"""The model boundary: shapes, dtypes, batch consistency and value semantics.

Covers Phase 5 gates 4 (`input_shape_and_dtype_validated`), 6
(`policy_output_contract_validated`) and part of 7
(`value_output_contract_validated`).
"""

from __future__ import annotations

import pytest
import torch

from stratego.engine.constants import OBSERVATION_VERSION, RULES_VERSION
from stratego.model.contract import (
    ACTION_ENCODING_VERSION,
    BELIEF_TYPE_COUNT,
    LEGACY_CONTRACT_V1,
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
    validate_belief_logits,
    validate_observation_batch,
    validate_policy_logits,
    validate_token_batch,
    validate_value_logits,
    value_probabilities,
)

from .conftest import deterministic_observation


# ---------------------------------------------------------------------------
# Frozen identifiers
# ---------------------------------------------------------------------------


def test_the_contract_names_the_frozen_upstream_versions():
    summary = contract_summary()
    assert summary["rules_version"] == RULES_VERSION == "stratego_project_v1"
    assert summary["observation_version"] == OBSERVATION_VERSION == "observation_v2_1_127ch"
    assert summary["action_encoding_version"] == ACTION_ENCODING_VERSION
    assert summary["model_contract_version"] == MODEL_CONTRACT_VERSION
    assert summary["action_index_rule"] == "action_id = 100 * source + destination"


def test_the_declared_shapes_are_the_phase_5_shapes():
    assert (TOKEN_COUNT, TOKEN_FEATURES) == (100, 127)
    assert POLICY_LOGIT_COUNT == 10_000
    assert BELIEF_TYPE_COUNT == 12
    assert VALUE_CLASS_ORDER == ("WIN", "DRAW", "LOSS")


def test_the_policy_frame_is_recorded_explicitly():
    # Under model_contract_v2 the tokens and the policy logits share one frame,
    # while the engine keeps its absolute identifiers. All three are asserted:
    # the whole point of v2 is that the first two agree and the third does not,
    # so an accidental "simplification" that collapsed them must fail here.
    assert POLICY_ACTION_FRAME == "perspective_normalized_squares"
    summary = contract_summary()
    assert summary["token_square_frame"] == "perspective_normalized_squares"
    assert summary["policy_action_frame"] == summary["token_square_frame"]
    assert summary["engine_action_frame"] == "absolute_engine_squares"
    assert summary["policy_action_frame"] != summary["engine_action_frame"]


def test_the_contract_version_moved_with_the_frame():
    # A frame change is a semantic change to every weight in the policy head, so
    # it must never ship under the version that meant something else.
    assert MODEL_CONTRACT_VERSION == "model_contract_v2"
    assert LEGACY_CONTRACT_V1["model_contract_version"] == "model_contract_v1"
    assert LEGACY_CONTRACT_V1["policy_action_frame"] == "absolute_engine_squares"
    # The action *encoding* is frozen and did not move with the frame.
    assert ACTION_ENCODING_VERSION == "source_destination_10000_v1"
    assert LEGACY_CONTRACT_V1["action_encoding_version"] == ACTION_ENCODING_VERSION


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_canonical_input_is_accepted():
    assert validate_observation_batch(deterministic_observation(batch=5)) == 5


@pytest.mark.parametrize(
    "shape",
    [
        (127, 10, 10),  # rank 3: a single observation is not a batch
        (1, 126, 10, 10),  # wrong channel count
        (1, 127, 10),  # rank 3 again, different failure
        (1, 127, 10, 9),  # wrong board width
        (1, 127, 9, 10),  # wrong board height
        (0, 127, 10, 10),  # empty batch
        (1, 10, 10, 127),  # channels-last: the classic mistake
    ],
)
def test_wrong_input_shapes_are_rejected(shape):
    with pytest.raises(ModelContractError):
        validate_observation_batch(torch.zeros(shape))


def test_integer_input_is_rejected_but_every_float_width_is_accepted():
    with pytest.raises(ModelContractError):
        validate_observation_batch(torch.zeros(1, 127, 10, 10, dtype=torch.int64))
    for dtype in (torch.float16, torch.float32, torch.float64):
        assert validate_observation_batch(torch.zeros(1, 127, 10, 10, dtype=dtype)) == 1


def test_non_finite_input_is_rejected():
    observation = deterministic_observation()
    observation[0, 3, 4, 5] = float("nan")
    with pytest.raises(ModelContractError, match="non-finite"):
        validate_observation_batch(observation)


def test_a_numpy_array_is_not_a_model_input():
    import numpy as np

    with pytest.raises(ModelContractError, match="torch.Tensor"):
        validate_observation_batch(np.zeros((1, 127, 10, 10), dtype=np.float32))


def test_token_batch_validation():
    assert validate_token_batch(torch.zeros(3, 100, 127)) == 3
    for bad in [(3, 127, 100), (3, 100, 126), (100, 127), (0, 100, 127)]:
        with pytest.raises(ModelContractError):
            validate_token_batch(torch.zeros(bad))


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def test_head_shapes_are_enforced():
    assert validate_policy_logits(torch.zeros(2, 10_000)) == 2
    assert validate_value_logits(torch.zeros(2, 3)) == 2
    assert validate_belief_logits(torch.zeros(2, 100, 12)) == 2

    with pytest.raises(ModelContractError):
        validate_policy_logits(torch.zeros(2, 9_999))
    with pytest.raises(ModelContractError):
        validate_value_logits(torch.zeros(2, 1))  # a scalar value head is refused
    with pytest.raises(ModelContractError):
        validate_belief_logits(torch.zeros(2, 12, 100))  # transposed belief head


def test_policy_logits_may_be_non_finite_but_value_and_belief_may_not():
    # The model is allowed to score an illegal index as -inf; the adapter is what
    # decides whether the *usable* entries are finite.
    validate_policy_logits(torch.full((1, 10_000), float("-inf")))
    with pytest.raises(ModelContractError):
        validate_policy_logits(torch.full((1, 10_000), float("nan")), require_finite=True)
    with pytest.raises(ModelContractError):
        validate_value_logits(torch.full((1, 3), float("inf")))
    with pytest.raises(ModelContractError):
        validate_belief_logits(torch.full((1, 100, 12), float("nan")))


def test_heads_must_agree_on_the_batch_dimension():
    with pytest.raises(ModelContractError, match="batch"):
        ModelOutputs.validated(torch.zeros(2, 10_000), torch.zeros(3, 3), torch.zeros(2, 100, 12))
    with pytest.raises(ModelContractError, match="batch"):
        ModelOutputs.validated(torch.zeros(2, 10_000), torch.zeros(2, 3), torch.zeros(1, 100, 12))


def test_validated_outputs_expose_rows_and_finiteness():
    outputs = ModelOutputs.validated(
        torch.zeros(4, 10_000), torch.zeros(4, 3), torch.zeros(4, 100, 12)
    )
    assert outputs.batch_size == 4
    assert outputs.all_finite()
    row = outputs.row(2)
    assert row.policy_logits.shape == (1, 10_000)
    assert row.value_logits.shape == (1, 3)
    assert row.belief_logits.shape == (1, 100, 12)


# ---------------------------------------------------------------------------
# Value semantics
# ---------------------------------------------------------------------------


def test_value_probabilities_sum_to_one_and_keep_class_order():
    logits = torch.tensor([[3.0, 0.0, -1.0], [-2.0, 5.0, 1.0]])
    probabilities = value_probabilities(logits)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    # Row 0 is a win-heavy position, row 1 a draw-heavy one.
    assert probabilities[0].argmax().item() == 0
    assert probabilities[1].argmax().item() == 1


def test_expected_value_is_win_minus_loss():
    logits = torch.tensor([[6.0, 0.0, -6.0], [-6.0, 0.0, 6.0], [0.0, 20.0, 0.0]])
    probabilities = value_probabilities(logits)
    manual = probabilities[:, 0] - probabilities[:, 2]
    assert torch.allclose(expected_value(logits), manual, atol=1e-7)
    # A confident win is near +1, a confident loss near -1, a certain draw near 0.
    assert expected_value(logits).tolist()[0] > 0.9
    assert expected_value(logits).tolist()[1] < -0.9
    assert abs(expected_value(logits).tolist()[2]) < 1e-3


def test_symmetric_win_loss_logits_give_zero_expected_value():
    assert abs(float(expected_value(torch.tensor([[1.5, 0.25, 1.5]]))[0])) < 1e-7
