"""Phase 15 Agent 1 sections 11-12: the metric block and the provider.

The metrics have arithmetic identities that must hold exactly — the
baseline scores `R_CE = 1` against itself, a perfect model scores zero
cross-entropy, a perfectly calibrated model scores zero ECE. The provider
has safety properties that must hold on real positions: probabilities on
the simplex, seed-reproducible worlds, exact remaining counts, no moved
piece assigned an immobile rank, and no path to a true rank.
"""

from __future__ import annotations

import numpy as np
import pytest

from stratego.belief.phase15 import contract as C
from stratego.belief.phase15 import metrics as M


def _fabricate(pieces_per_sample=(2, 3, 1), games=(0, 0, 1)):
    total = sum(pieces_per_sample)
    offsets = np.concatenate([[0], np.cumsum(pieces_per_sample)]).astype(np.int64)
    rng = np.random.default_rng(7)
    counts = np.full((len(pieces_per_sample), 12), 3, dtype=np.int16)
    return {
        "split": "development",
        "samples": len(pieces_per_sample),
        "pieces": total,
        "games": len(set(games)),
        "piece_offset": offsets,
        "game_ordinal": np.asarray(games, dtype=np.int32),
        "observer_model": np.zeros(len(pieces_per_sample), dtype=np.int8),
        "opponent": np.zeros(len(pieces_per_sample), dtype=np.int8),
        "setup_source": np.zeros(len(pieces_per_sample), dtype=np.int8),
        "observer_family": np.zeros(len(pieces_per_sample), dtype=np.int8),
        "opponent_family": np.zeros(len(pieces_per_sample), dtype=np.int8),
        "observer_color": np.zeros(len(pieces_per_sample), dtype=np.int8),
        "total_moves": np.array([10, 80, 200], dtype=np.int32)[
            : len(pieces_per_sample)
        ],
        "remaining_counts": counts,
        "legal_rank_mask": np.ones((total, 12), dtype=bool),
        "piece_moved": np.zeros(total, dtype=bool),
        "perspective_square": np.arange(total, dtype=np.int16),
        "true_rank": rng.integers(0, 12, size=total).astype(np.int8),
    }


# ---------------------------------------------------------------------------
# Arithmetic identities
# ---------------------------------------------------------------------------


def test_the_baseline_scores_exactly_one_against_itself():
    data = _fabricate()
    baseline = M.baseline_probabilities(data)
    block = M.evaluate(data=data, probabilities=baseline, bootstrap_resamples=50)
    assert block["r_ce"] == pytest.approx(1.0)
    assert block["ce"] == pytest.approx(block["baseline_ce"])


def test_a_uniform_mask_and_equal_counts_give_a_uniform_baseline():
    data = _fabricate()
    baseline = M.baseline_probabilities(data)
    assert np.allclose(baseline, 1.0 / 12.0)


def test_a_perfect_model_scores_zero_cross_entropy_and_zero_brier():
    data = _fabricate()
    perfect = np.zeros((data["pieces"], 12))
    perfect[np.arange(data["pieces"]), np.asarray(data["true_rank"], dtype=int)] = 1.0
    assert M.cross_entropy(perfect, data["true_rank"]).max() == pytest.approx(0.0)
    assert M.brier(perfect, data["true_rank"]).max() == pytest.approx(0.0)


def test_brier_is_the_multiclass_form():
    probabilities = np.array([[0.7, 0.3] + [0.0] * 10])
    assert M.brier(probabilities, np.array([0])) == pytest.approx(0.09 + 0.09)


def test_a_zero_mass_row_is_clamped_and_counted():
    data = _fabricate()
    probabilities = np.full((data["pieces"], 12), 1.0 / 11.0)
    probabilities[:, int(data["true_rank"][0])] = 0.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    block = M.evaluate(
        data=data, probabilities=probabilities, bootstrap_resamples=20,
        with_breakdowns=False,
    )
    assert block["zero_mass_rows"] >= 1
    assert np.isfinite(block["ce"])


def test_calibration_error_is_zero_for_a_perfectly_calibrated_predictor():
    rng = np.random.default_rng(1)
    rows = 40000
    probabilities = np.full((rows, 12), 1.0 / 12.0)
    true_rank = rng.integers(0, 12, size=rows)
    report = M.calibration_error(probabilities, true_rank)
    # Every row has confidence 1/12 and accuracy 1/12 in expectation.
    assert report["expected_calibration_error"] < 0.01
    assert report["maximum_calibration_error"] < 0.01


def test_calibration_error_is_large_for_a_confidently_wrong_predictor():
    rows = 4000
    probabilities = np.full((rows, 12), 0.01 / 11.0)
    probabilities[:, 0] = 0.99
    true_rank = np.ones(rows, dtype=np.int64)
    report = M.calibration_error(probabilities, true_rank)
    assert report["expected_calibration_error"] > 0.9
    assert report["maximum_calibration_error"] > 0.9


def test_evaluate_refuses_probabilities_of_the_wrong_shape():
    data = _fabricate()
    with pytest.raises(M.Phase15MetricsError):
        M.evaluate(data=data, probabilities=np.zeros((3, 12)))
    with pytest.raises(M.Phase15MetricsError):
        M.evaluate(data=data, probabilities=np.full((data["pieces"], 12), np.nan))


def test_evaluate_refuses_a_split_without_labels():
    data = _fabricate()
    data.pop("true_rank")
    with pytest.raises(M.Phase15MetricsError):
        M.evaluate(data=data, probabilities=np.full((6, 12), 1 / 12))


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------


def test_every_required_breakdown_dimension_is_present():
    data = _fabricate()
    block = M.evaluate(
        data=data, probabilities=M.baseline_probabilities(data), bootstrap_resamples=20
    )
    for dimension in (
        "observer_color",
        "observer_source",
        "opponent",
        "opponent_class",
        "setup_source",
        "opponent_setup_family",
        "game_band",
    ):
        assert dimension in block["breakdowns"]


def test_the_bands_are_cut_at_the_declared_plies():
    assert C.game_band(0) == "early"
    assert C.game_band(C.GAME_BAND_EARLY_MAX_PLY) == "early"
    assert C.game_band(C.GAME_BAND_EARLY_MAX_PLY + 1) == "middle"
    assert C.game_band(C.GAME_BAND_MIDDLE_MAX_PLY) == "middle"
    assert C.game_band(C.GAME_BAND_MIDDLE_MAX_PLY + 1) == "late"
    with pytest.raises(C.Phase15Error):
        C.game_band(-1)


def test_piece_level_views_expand_per_sample_labels_correctly():
    data = _fabricate(pieces_per_sample=(2, 3, 1))
    assert list(M.piece_samples(data)) == [0, 0, 1, 1, 1, 2]
    assert list(M.piece_games(data)) == [0, 0, 0, 0, 0, 1]
    assert [
        C.GAME_BANDS[index] for index in M.piece_bands(data)
    ] == ["early", "early", "middle", "middle", "middle", "late"]


def test_a_paired_comparison_of_a_model_with_itself_is_zero():
    data = _fabricate()
    baseline = M.baseline_probabilities(data)
    paired = M.paired_comparison(baseline, baseline, data, resamples=50)
    assert paired["ce_difference"] == pytest.approx(0.0)
    assert paired["distinguishable"] is False


def test_the_uniform_reference_is_worse_than_the_count_baseline():
    data = _fabricate()
    counts = np.asarray(data["remaining_counts"]).copy()
    counts[:, 0] = 20  # make the baseline genuinely informative
    data["remaining_counts"] = counts
    data["true_rank"] = np.zeros(data["pieces"], dtype=np.int8)
    reference = M.uniform_reference(data)
    assert reference["r_ce"] > 1.0


# ---------------------------------------------------------------------------
# The provider (section 12)
# ---------------------------------------------------------------------------


def test_the_provider_public_state_type_is_the_accepted_one():
    from stratego.belief.phase11b.interface import Phase11BPublicState
    from stratego.belief.phase15.interface import Phase15PublicState

    assert Phase15PublicState is Phase11BPublicState


def test_the_provider_refuses_a_state_that_is_not_a_public_state():
    from stratego.belief.phase15.interface import (
        Phase15BeliefProvider,
        Phase15InterfaceError,
    )

    provider = Phase15BeliefProvider.__new__(Phase15BeliefProvider)
    with pytest.raises(Phase15InterfaceError):
        Phase15BeliefProvider.predict_marginals(provider, {"observer_color": "red"})


def test_the_public_state_type_refuses_a_foreign_document():
    from stratego.belief.phase15.interface import Phase15PublicState
    from stratego.belief.phase11b.contract import Phase11BError

    with pytest.raises(Phase11BError):
        Phase15PublicState(
            public_state_document={"document_version": "not_ours"},
            observation=np.zeros(C.OBSERVATION_SHAPE, dtype=np.float32),
        )


def test_the_public_state_type_has_no_field_a_true_rank_could_arrive_in():
    from stratego.belief.phase15.interface import Phase15PublicState

    assert set(Phase15PublicState.__dataclass_fields__) == {
        "public_state_document",
        "observation",
    }
