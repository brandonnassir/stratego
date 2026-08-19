"""Phase 11 Agent 5: the independent recomputation path.

The point of `phase11_recompute` is that it does **not** call the code it
audits, so these tests do two things: they pin every restated constant to
the live contract value (a drift there must fail loudly rather than
propagate to both paths), and they show the independent arithmetic agrees
with the accepted evaluator on constructed data while still disagreeing
when the inputs really differ.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stratego.evaluation import phase11_evaluator as evaluator
from stratego.evaluation import phase11_recompute as recompute
from stratego.training import phase11_contract as contract
from stratego.training import phase11_seed as seed


# ---------------------------------------------------------------------------
# The restated constants must equal the frozen ones
# ---------------------------------------------------------------------------


def test_restated_constants_match_the_contract():
    assert recompute._RANKS == contract.RANK_COUNT
    assert recompute._LOG_FLOOR == contract.LOG_PROBABILITY_FLOOR
    assert recompute._ECE_BINS == int(contract.ECE_SPECIFICATION["bins"])
    assert recompute._REPLICATES == contract.BOOTSTRAP_REPLICATES
    assert recompute._CONFIDENCE == contract.BOOTSTRAP_CONFIDENCE
    assert recompute._STRATA == contract.OPPONENT_STRATA


def test_restated_seed_material_matches_the_frozen_seeds():
    assert recompute._IDENTITY_VERSION == seed.PHASE11_IDENTITY_VERSION
    assert recompute._PERSON == seed._PHASE11_SEED_PERSON
    assert recompute._BOOTSTRAP_DOMAIN == seed.DOMAIN_BOOTSTRAP
    assert recompute._BOOTSTRAP_DOMAIN_ROOT == seed.DOMAIN_ROOTS[seed.DOMAIN_BOOTSTRAP]
    assert recompute._BANK_BOOTSTRAP_ROOT == {
        "validation": seed.VALIDATION_BOOTSTRAP_SEED,
        "test": seed.TEST_BOOTSTRAP_SEED,
    }


def test_recompute_version_and_tolerance_are_frozen():
    assert recompute.RECOMPUTE_VERSION == "phase11_independent_recompute_v1"
    assert recompute.RECOMPUTE_TOLERANCE == 1e-9


# ---------------------------------------------------------------------------
# The re-derived stream seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bank", ["validation", "test"])
@pytest.mark.parametrize(
    "token",
    [
        "ce_learned",
        "ce_delta",
        "r_ce",
        "top1_delta",
        "brier_delta",
        "r_ce|st=basic_rule",
        "ce_learned|st=scout_rush",
    ],
)
def test_independent_bootstrap_seed_reproduces_the_frozen_stream(bank, token):
    assert recompute.independent_bootstrap_seed(bank, token) == (
        seed.bootstrap_stream_seed(bank, token)
    )


def test_independent_bootstrap_seed_refuses_an_unknown_bank():
    with pytest.raises(recompute.Phase11RecomputeError):
        recompute.independent_bootstrap_seed("holdout", "ce_learned")


def test_distinct_tokens_get_distinct_streams():
    seeds = {
        recompute.independent_bootstrap_seed("validation", token)
        for token in ("ce_learned", "ce_baseline", "ce_delta", "r_ce")
    }
    assert len(seeds) == 4


# ---------------------------------------------------------------------------
# The re-implemented quantile
# ---------------------------------------------------------------------------


def test_independent_quantile_matches_the_accepted_one():
    from stratego.evaluation.statistics import quantile

    values = sorted(float(value) for value in np.linspace(-3.0, 7.0, 257))
    for probability in (0.0, 0.025, 0.1, 0.5, 0.975, 1.0):
        assert recompute.independent_quantile(values, probability) == pytest.approx(
            quantile(values, probability), abs=1e-12
        )


def test_independent_quantile_handles_a_single_sample():
    assert recompute.independent_quantile([1.5], 0.3) == 1.5


def test_independent_quantile_refuses_an_empty_sample():
    with pytest.raises(recompute.Phase11RecomputeError):
        recompute.independent_quantile([], 0.5)


# ---------------------------------------------------------------------------
# Per-event scores by the other arithmetic route
# ---------------------------------------------------------------------------


def _logits(rows: int, seed_value: int = 11) -> np.ndarray:
    generator = np.random.default_rng(seed_value)
    return generator.normal(0.0, 2.5, size=(rows, contract.RANK_COUNT)).astype(
        np.float32
    )


def test_learned_scores_agree_with_the_evaluator():
    logits = _logits(512)
    true_rank = np.random.default_rng(3).integers(
        0, contract.RANK_COUNT, size=logits.shape[0]
    )
    probabilities = np.stack(
        [evaluator_softmax(row) for row in logits]
    )
    primary = evaluator.score_matrix(probabilities, true_rank)
    independent = recompute.independent_learned_scores(logits, true_rank)
    for name in ("ce", "top1", "brier", "entropy", "true_rank_probability", "confidence"):
        assert np.abs(primary[name] - independent[name]).max() < 1e-12


def evaluator_softmax(row):
    from stratego.evaluation.phase11_belief import softmax_float64

    return softmax_float64(row)


def test_learned_ce_respects_the_frozen_probability_floor():
    """A rank the head assigns ~zero mass must hit the floored CE, not inf."""
    logits = np.zeros((1, contract.RANK_COUNT), dtype=np.float32)
    logits[0, 0] = 200.0
    scores = recompute.independent_learned_scores(logits, np.array([5]))
    assert scores["ce"][0] == pytest.approx(-math.log(contract.LOG_PROBABILITY_FLOOR))
    assert math.isfinite(scores["ce"][0])


def test_learned_scores_refuse_a_wrong_rank_width():
    with pytest.raises(recompute.Phase11RecomputeError):
        recompute.independent_learned_scores(np.zeros((2, 7)), np.array([0, 1]))


def test_baseline_scores_agree_with_the_accepted_baseline():
    from stratego.evaluation.phase11_baselines import remaining_count_distribution

    generator = np.random.default_rng(5)
    counts = generator.integers(0, 4, size=(64, contract.RANK_COUNT)).astype(np.float64)
    counts[:, 3] += 1  # keep every row's legal mass positive
    masks = np.ones_like(counts)
    masks[:32, 10] = 0.0
    masks[:32, 11] = 0.0
    true_rank = np.array(
        [int(np.flatnonzero(counts[row] * masks[row])[0]) for row in range(64)]
    )
    probabilities = np.stack(
        [remaining_count_distribution(counts[row], masks[row]) for row in range(64)]
    )
    primary = evaluator.score_matrix(probabilities, true_rank)
    independent = recompute.independent_baseline_scores(counts, masks, true_rank)
    for name in ("ce", "top1", "brier", "entropy", "true_rank_probability", "confidence"):
        assert np.abs(primary[name] - independent[name]).max() < 1e-12


def test_baseline_scores_refuse_a_row_with_no_legal_mass():
    counts = np.zeros((1, contract.RANK_COUNT))
    masks = np.ones((1, contract.RANK_COUNT))
    with pytest.raises(recompute.Phase11RecomputeError):
        recompute.independent_baseline_scores(counts, masks, np.array([0]))


# ---------------------------------------------------------------------------
# Aggregation, bootstrap and ECE
# ---------------------------------------------------------------------------


def test_case_means_pool_both_games_of_a_case():
    values = np.array([1.0, 3.0, 10.0, 10.0])
    case_index = np.array([0, 0, 1, 1])
    present, means = recompute.independent_case_means(values, case_index, 2)
    assert present == [0, 1]
    assert means == pytest.approx([2.0, 10.0])


def test_case_means_skip_a_case_with_no_events():
    values = np.array([1.0, 3.0])
    case_index = np.array([0, 0])
    present, means = recompute.independent_case_means(values, case_index, 3)
    assert present == [0]
    assert means == pytest.approx([2.0])


def test_bootstrap_mean_matches_the_accepted_bootstrap():
    values = np.random.default_rng(7).normal(2.0, 0.5, size=512)
    primary = evaluator.bootstrap_mean(values, "validation", "ce_learned")
    independent = recompute.independent_bootstrap_mean(
        values, "validation", "ce_learned"
    )
    assert independent["stream_seed"] == primary["stream_seed"]
    for key in ("point", "lower", "upper"):
        assert independent[key] == pytest.approx(primary[key], abs=1e-12)


def test_bootstrap_ratio_matches_the_accepted_bootstrap():
    generator = np.random.default_rng(9)
    numerator = generator.normal(2.1, 0.3, size=512)
    denominator = generator.normal(2.2, 0.3, size=512)
    primary = evaluator.bootstrap_ratio(
        numerator, denominator, "validation", "r_ce"
    )
    independent = recompute.independent_bootstrap_ratio(
        numerator, denominator, "validation", "r_ce"
    )
    assert independent["stream_seed"] == primary["stream_seed"]
    for key in ("point", "lower", "upper"):
        assert independent[key] == pytest.approx(primary[key], abs=1e-12)


def test_bootstrap_ratio_refuses_mismatched_aggregates():
    with pytest.raises(recompute.Phase11RecomputeError):
        recompute.independent_bootstrap_ratio([1.0, 2.0], [1.0], "validation", "r_ce")


def test_empty_bootstrap_returns_nan_without_a_stream():
    block = recompute.independent_bootstrap_mean([], "validation", "ce_learned")
    assert math.isnan(block["point"])
    assert block["cases"] == 0
    assert "stream_seed" not in block


def test_independent_ece_matches_the_accepted_ece():
    generator = np.random.default_rng(13)
    confidence = generator.uniform(0.0, 1.0, size=4096)
    correct = (generator.uniform(size=4096) < confidence).astype(np.float64)
    primary = evaluator.expected_calibration_error(confidence, correct)["ece"]
    assert recompute.independent_ece(confidence, correct) == pytest.approx(
        primary, abs=1e-12
    )


def test_independent_ece_puts_confidence_one_in_the_last_bin():
    assert recompute.independent_ece([1.0], [1.0]) == pytest.approx(0.0)
    assert recompute.independent_ece([1.0], [0.0]) == pytest.approx(1.0)


def test_independent_ece_of_no_events_is_nan():
    assert math.isnan(recompute.independent_ece([], []))


# ---------------------------------------------------------------------------
# Comparison semantics
# ---------------------------------------------------------------------------


def test_deviation_treats_a_one_sided_nan_as_total_disagreement():
    assert recompute._deviation(float("nan"), 1.0) == float("inf")
    assert recompute._deviation(1.0, float("nan")) == float("inf")
    assert recompute._deviation(float("nan"), float("nan")) == 0.0
    assert recompute._deviation(1.0, 1.25) == pytest.approx(0.25)


def test_deviation_records_nan_pairs_when_asked():
    seen: list = []
    recompute._deviation(float("nan"), float("nan"), seen)
    recompute._deviation(1.0, 1.0, seen)
    assert len(seen) == 1


def test_recompute_bank_requires_an_explicit_shard_reader():
    with pytest.raises(recompute.Phase11RecomputeError):
        recompute.recompute_bank(".", {"games_index": []}, "validation")


# ---------------------------------------------------------------------------
# The module really is independent
# ---------------------------------------------------------------------------


def test_recompute_module_imports_no_phase11_implementation():
    """The audit must not call the code it audits."""
    import ast
    from pathlib import Path

    source = Path(recompute.__file__).read_text()
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if "phase11" in name], imported
