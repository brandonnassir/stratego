"""Phase 11 Agent 2: the independent recomputation and the negative controls."""

import math

import numpy as np
import pytest

from stratego.engine.constants import RED
from stratego.engine.observation import build_observation
from stratego.engine.state import create_game
from stratego.evaluation.match_spec import EVALUATION_RULES
from stratego.evaluation.phase11_audit import (
    AUDIT_TOLERANCE,
    BASELINE_EDGE_CASES,
    NEGATIVE_CONTROLS,
    compare_scores,
    independent_scores,
    independent_softmax,
    run_negative_controls,
    scalar_event_metrics,
    scalar_recompute,
)
from stratego.evaluation.phase11_belief import softmax_float64
from stratego.evaluation.phase11_evaluator import score_matrix
from stratego.evaluation.phase11_public_state import build_public_state_document
from stratego.evaluation.policy import build_public_view
from stratego.setups.contracts import LIBRARY_JSONL_PATH
from stratego.setups.library import read_library_jsonl


@pytest.fixture(scope="module")
def position():
    entries = read_library_jsonl(LIBRARY_JSONL_PATH)
    state = create_game(
        tuple(entries[0].canonical_setup),
        tuple(entries[1].canonical_setup),
        rules=EVALUATION_RULES,
        game_id="phase11-audit",
    )
    observation = build_observation(state, RED)
    document = build_public_state_document(build_public_view(state, RED), observation)
    return document, observation


@pytest.fixture(scope="module")
def sample(position):
    document, observation = position
    rng = np.random.default_rng(31)
    probabilities = rng.dirichlet(np.ones(12) * 0.7, size=256)
    return {
        "probabilities": probabilities,
        "true_rank": rng.integers(0, 12, size=256),
        "document": document,
        "observation": observation,
    }


# ---------------------------------------------------------------------------
# Layer 1 — the independent formulas
# ---------------------------------------------------------------------------


def test_the_two_implementations_agree_on_random_rows(sample):
    primary = score_matrix(sample["probabilities"], sample["true_rank"])
    audit = independent_scores(sample["probabilities"], sample["true_rank"])
    comparison = compare_scores(primary, audit)
    assert comparison["within_tolerance"], comparison["max_deviation"]


def test_the_two_softmaxes_agree_on_ordinary_logits():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(64, 12)).astype(np.float32)
    shifted = np.stack([softmax_float64(row) for row in logits])
    plain = independent_softmax(logits.astype(np.float64))
    assert float(np.abs(shifted - plain).max()) < 1e-12


def test_the_audit_formulas_are_actually_different():
    """A tautological audit would agree even on a corrupted input."""
    rng = np.random.default_rng(9)
    probabilities = rng.dirichlet(np.ones(12), size=32)
    truth = rng.integers(0, 12, size=32)
    primary = score_matrix(probabilities, truth)
    audit = independent_scores(np.roll(probabilities, 1, axis=1), truth)
    assert not compare_scores(primary, audit)["within_tolerance"]


# ---------------------------------------------------------------------------
# Layer 2 — the scalar path
# ---------------------------------------------------------------------------


def test_scalar_event_metrics_match_the_vectorised_ones(sample):
    primary = score_matrix(sample["probabilities"], sample["true_rank"])
    for index in range(0, 256, 37):
        row = sample["probabilities"][index].tolist()
        scalar = scalar_event_metrics(row, int(sample["true_rank"][index]))
        for name in ("ce", "top1", "brier", "entropy", "true_rank_probability"):
            assert scalar[name] == pytest.approx(
                float(primary[name][index]), abs=AUDIT_TOLERANCE
            )


def test_scalar_recompute_builds_case_aggregates():
    records = [
        {
            "case_id": "c0",
            "learned_probabilities": (np.ones(12) / 12).tolist(),
            "baseline_probabilities": (np.ones(12) / 12).tolist(),
            "true_rank_index": 0,
        },
        {
            "case_id": "c0",
            "learned_probabilities": [1.0] + [0.0] * 11,
            "baseline_probabilities": (np.ones(12) / 12).tolist(),
            "true_rank_index": 0,
        },
        {
            "case_id": "c1",
            "learned_probabilities": (np.ones(12) / 12).tolist(),
            "baseline_probabilities": (np.ones(12) / 12).tolist(),
            "true_rank_index": 5,
        },
    ]
    result = scalar_recompute(records)
    assert result["events"] == 3
    assert result["cases"] == 2
    assert result["case_aggregates"]["c0"]["ce_learned"] == pytest.approx(
        (math.log(12.0) + 0.0) / 2
    )
    assert result["case_aggregates"]["c1"]["ce_learned"] == pytest.approx(math.log(12.0))
    # equal case weight: not the pooled mean of the three events
    assert result["overall"]["ce_learned"] == pytest.approx(
        (math.log(12.0) / 2 + math.log(12.0)) / 2
    )
    assert result["overall"]["r_ce"] == pytest.approx(
        result["overall"]["ce_learned"] / result["overall"]["ce_baseline"]
    )


def test_scalar_recompute_refuses_unscored_records():
    from stratego.evaluation.phase11_audit import Phase11AuditError

    with pytest.raises(Phase11AuditError):
        scalar_recompute(
            [
                {
                    "case_id": "c",
                    "learned_probabilities": [1.0] + [0.0] * 11,
                    "baseline_probabilities": [1.0] + [0.0] * 11,
                    "true_rank_index": None,
                }
            ]
        )


# ---------------------------------------------------------------------------
# The negative controls
# ---------------------------------------------------------------------------


def test_every_negative_control_fires_on_the_opening(sample):
    result = run_negative_controls(sample)
    fired = {control["control"]: control["fired"] for control in result["controls"]}
    assert tuple(fired) == NEGATIVE_CONTROLS
    # The opening has no revealed opponent piece, so control 4 has nothing to
    # find there and is exercised on a reveal position by the harness.
    expected = set(NEGATIVE_CONTROLS) - {"known_pieces_in_hidden_denominator"}
    assert all(fired[name] for name in expected), fired


def test_the_known_piece_control_fires_once_a_rank_is_revealed(sample):
    document = sample["document"]
    pieces = [dict(piece) for piece in document["pieces"]]
    revealed = 0
    for piece in pieces:
        if piece["owner_color"] == "blue" and revealed < 2:
            piece["known_to_observer"] = True
            piece["known_rank_index"] = 1
            revealed += 1
    polluted = {**document, "pieces": pieces}
    result = run_negative_controls({**sample, "document": polluted})
    fired = {control["control"]: control["fired"] for control in result["controls"]}
    assert fired["known_pieces_in_hidden_denominator"]
    assert all(fired.values()), fired


def test_the_truth_injection_control_names_the_refusal(sample):
    result = run_negative_controls(sample)
    control = next(
        item
        for item in result["controls"]
        if item["control"] == "hidden_truth_injected_into_request"
    )
    assert control["fired"]
    assert "allowlist" in control["detail"]["refusal"]


def test_the_frozen_edge_case_list_is_the_instruction_list():
    assert BASELINE_EDGE_CASES == (
        "moved_unknown",
        "revealed_rank",
        "capture",
        "near_endgame_exhaustion",
        "single_legal_rank",
        "public_scout_deduction",
    )
