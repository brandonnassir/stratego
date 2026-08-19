"""Phase 11 Agent 2: the production belief request boundary and extraction."""

import numpy as np
import pytest

from stratego.engine.constants import RED
from stratego.engine.observation import build_observation
from stratego.engine.state import create_game
from stratego.evaluation.match_spec import EVALUATION_RULES
from stratego.evaluation.phase11_belief import (
    PLAYER_BY_COLOR,
    Phase11BeliefError,
    Phase11BeliefPrediction,
    Phase11BeliefRequest,
    softmax_float64,
)
from stratego.evaluation.phase11_public_state import build_public_state_document
from stratego.evaluation.policy import build_public_view
from stratego.setups.contracts import LIBRARY_JSONL_PATH
from stratego.setups.library import read_library_jsonl
from stratego.training.phase11_contract import (
    ALLOWED_BELIEF_REQUEST_FIELDS,
    BELIEF_REQUEST_VERSION,
    FORBIDDEN_BELIEF_REQUEST_TOKENS,
)


@pytest.fixture(scope="module")
def position():
    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    state = create_game(
        tuple(entries[0].canonical_setup),
        tuple(entries[1].canonical_setup),
        rules=EVALUATION_RULES,
        game_id="phase11-belief",
    )
    observation = build_observation(state, RED)
    document = build_public_state_document(build_public_view(state, RED), observation)
    return document, observation


@pytest.fixture(scope="module")
def payload(position):
    document, observation = position
    return {
        "request_version": BELIEF_REQUEST_VERSION,
        "request_id": "test#0",
        "observer_color": "red",
        "public_state_document": document,
        "observation": observation,
    }


def test_a_well_formed_payload_builds(payload):
    request = Phase11BeliefRequest.from_payload(payload)
    assert request.observer == PLAYER_BY_COLOR["red"]
    assert request.public_state_identity
    assert request.digest() == Phase11BeliefRequest.from_payload(payload).digest()


@pytest.mark.parametrize(
    "field",
    [
        "true_rank_index",
        "truth",
        "hidden_labels",
        "target_ranks",
        "private_pieces",
        "winner",
        "result",
        "reward",
        "outcome",
        "future_actions",
        "storage_path",
        "opponent_setup",
    ],
)
def test_an_off_allowlist_field_is_refused_not_dropped(payload, field):
    with pytest.raises(Phase11BeliefError) as error:
        Phase11BeliefRequest.from_payload({**payload, field: "anything"})
    assert "allowlist" in str(error.value)


def test_every_forbidden_token_is_covered_by_the_allowlist():
    for token in FORBIDDEN_BELIEF_REQUEST_TOKENS:
        assert not any(token in field for field in ALLOWED_BELIEF_REQUEST_FIELDS)


def test_a_forbidden_field_inside_the_document_is_refused(payload):
    polluted = {**payload["public_state_document"], "true_ranks": [1, 2, 3]}
    with pytest.raises(Phase11BeliefError) as error:
        Phase11BeliefRequest.from_payload(
            {**payload, "public_state_document": polluted}
        )
    assert "forbidden" in str(error.value)


def test_a_missing_field_is_refused(payload):
    for field in ALLOWED_BELIEF_REQUEST_FIELDS:
        partial = {key: value for key, value in payload.items() if key != field}
        with pytest.raises(Phase11BeliefError):
            Phase11BeliefRequest.from_payload(partial)


def test_the_request_fields_are_exactly_the_frozen_allowlist():
    from dataclasses import fields

    assert tuple(
        item.name for item in fields(Phase11BeliefRequest)
    ) == ALLOWED_BELIEF_REQUEST_FIELDS


def test_a_mismatched_observation_is_refused(payload):
    other = np.zeros((127, 10, 10), dtype=np.float32)
    with pytest.raises(Phase11BeliefError) as error:
        Phase11BeliefRequest.from_payload({**payload, "observation": other})
    assert "observation_sha256" in str(error.value)


def test_a_disagreeing_observer_colour_is_refused(payload):
    with pytest.raises(Phase11BeliefError):
        Phase11BeliefRequest.from_payload({**payload, "observer_color": "blue"})


def test_a_foreign_request_version_is_refused(payload):
    with pytest.raises(Phase11BeliefError):
        Phase11BeliefRequest.from_payload({**payload, "request_version": "v9"})


def test_a_non_mapping_payload_is_refused():
    with pytest.raises(Phase11BeliefError):
        Phase11BeliefRequest.from_payload(["not", "a", "mapping"])


# ---------------------------------------------------------------------------
# The frozen probability extraction
# ---------------------------------------------------------------------------


def test_softmax_is_a_full_simplex_with_no_masking():
    logits = np.array([-2.0, 0.0, 1.5, 3.0, -7.0, 0.1, 0.2, 0.3, 0.4, 0.5, 9.0, -9.0])
    row = softmax_float64(logits.astype(np.float32))
    assert row.dtype == np.float64
    assert row.sum() == pytest.approx(1.0, abs=1e-15)
    assert (row > 0.0).all()
    assert row.argmax() == 10


def test_softmax_is_stable_on_a_large_logit():
    row = softmax_float64(np.full(12, 800.0, dtype=np.float32))
    assert np.isfinite(row).all()
    assert row.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(row, 1.0 / 12.0)


def test_softmax_refuses_a_wrong_shape_or_non_finite_row():
    with pytest.raises(Phase11BeliefError):
        softmax_float64(np.zeros(11, dtype=np.float32))
    with pytest.raises(Phase11BeliefError):
        softmax_float64(np.full(12, np.nan, dtype=np.float32))


def test_a_prediction_carries_no_truth_field():
    from dataclasses import fields

    names = {item.name for item in fields(Phase11BeliefPrediction)}
    assert not names & {"true_rank_index", "true_rank", "truth", "target"}
    prediction = Phase11BeliefPrediction(
        request_id="r",
        observer_color="red",
        public_state_identity="a" * 64,
        observation_sha256="b" * 64,
        total_moves=0,
        belief_logits={0: np.zeros(12, dtype=np.float32)},
        perspective_squares={0: 5},
    )
    assert prediction.probabilities()[0].sum() == pytest.approx(1.0, abs=1e-15)
