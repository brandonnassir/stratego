"""Regression: the frozen Phase 10 feature order, standardizer and fit protocol.

Agent 1 fits nothing. What is pinned here is the *definition* Agent 3 must
implement without a remaining choice: the lossless 47-scalar flattening of
the 35 frozen trait fields, the train-only standardizer and its digest, the
parameter layout, and the exact objective.
"""

import numpy as np
import pytest

from stratego.setups.contracts import LIBRARY_JSONL_PATH
from stratego.setups.library import read_library_jsonl
from stratego.setups.traits import TRAIT_SCHEMA, compute_trait_vector
from stratego.training import phase10_utility as pu
from tests.training.phase10_frozen_digests import TRAIT_SCALER_DIGEST


@pytest.fixture(scope="module")
def scaler():
    return pu.fit_trait_scaler()


class TestFeatureFlattening:
    def test_the_source_schema_has_35_fields(self):
        assert len(TRAIT_SCHEMA) == 35

    def test_the_flattening_is_lossless_and_47_wide(self):
        assert pu.TRAIT_FEATURE_COUNT == 47
        expanded = sum(
            4 if field.kind == "int_list4" else 1 for field in TRAIT_SCHEMA
        )
        assert expanded == 47

    def test_feature_order_follows_the_schema(self):
        assert pu.TRAIT_FEATURE_NAMES[0] == "flag_rank"
        assert "bomb_rank_histogram[0]" in pu.TRAIT_FEATURE_NAMES
        assert "bomb_rank_histogram[3]" in pu.TRAIT_FEATURE_NAMES
        assert pu.TRAIT_FEATURE_NAMES[-1] == "unconventional_feature_count"
        assert len(set(pu.TRAIT_FEATURE_NAMES)) == 47

    def test_a_feature_vector_reproduces_its_trait_vector(self):
        entry = read_library_jsonl(LIBRARY_JSONL_PATH)[0]
        vector = pu.trait_feature_vector(entry.trait_vector)
        assert len(vector) == 47
        assert vector == pu.trait_feature_vector(compute_trait_vector(entry.canonical_setup))

    def test_a_missing_field_is_refused(self):
        entry = read_library_jsonl(LIBRARY_JSONL_PATH)[0]
        damaged = dict(entry.trait_vector)
        damaged.pop("flag_rank")
        with pytest.raises(pu.Phase10UtilityError):
            pu.trait_feature_vector(damaged)

    def test_the_single_non_invariant_field_is_recorded(self):
        assert pu.NON_REFLECTION_INVARIANT_FEATURES == ("flag_file",)


class TestScaler:
    def test_the_scaler_sees_train_bases_only(self, scaler):
        assert scaler.base_count == 6400
        assert scaler.split == "train"
        base_ids, families, features = pu.load_train_features()
        assert len(base_ids) == 6400
        assert features.shape == (6400, 47)
        assert len(set(families)) == 16

    def test_population_convention_is_ddof_zero(self, scaler):
        _, _, features = pu.load_train_features()
        np.testing.assert_allclose(scaler.std, features.std(axis=0, ddof=0))
        np.testing.assert_allclose(scaler.mean, features.mean(axis=0))

    def test_transform_standardizes_the_train_population(self, scaler):
        _, _, features = pu.load_train_features()
        standardized = scaler.transform(features)
        active = scaler.std != 0.0
        np.testing.assert_allclose(standardized[:, active].mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(standardized[:, active].std(axis=0, ddof=0), 1.0, atol=1e-10)

    def test_a_zero_std_feature_standardizes_to_zero(self):
        mean = np.zeros(47)
        std = np.zeros(47)
        std[0] = 2.0
        degenerate = pu.TraitScaler(mean, std, base_count=1)
        assert degenerate.zero_std_features == pu.TRAIT_FEATURE_NAMES[1:]
        transformed = degenerate.transform(np.ones(47))
        assert transformed[0] == 0.5
        assert np.all(transformed[1:] == 0.0)

    def test_scaler_digest_is_pinned_and_stable(self, scaler):
        assert scaler.digest() == TRAIT_SCALER_DIGEST
        assert pu.fit_trait_scaler().digest() == TRAIT_SCALER_DIGEST


class TestFitProtocol:
    def test_parameter_layouts(self):
        family_only = pu.parameter_layout("model_F")
        with_traits = pu.parameter_layout("model_T")
        assert family_only["total_parameters"] == 33
        assert with_traits["total_parameters"] == 127
        assert family_only["trait_weights"] is None
        assert with_traits["trait_weights"]["shape"] == [2, 47]

    def test_unknown_model_is_refused(self):
        with pytest.raises(pu.Phase10UtilityError):
            pu.parameter_layout("model_X")

    def test_frozen_protocol_values(self):
        assert pu.FIT_PROTOCOL["device"] == "cpu"
        assert pu.FIT_PROTOCOL["precision"] == "float64"
        assert pu.FIT_PROTOCOL["max_iterations"] == 500
        assert pu.FIT_PROTOCOL["history_size"] == 50
        assert pu.FIT_PROTOCOL["tolerance_grad"] == 1e-10
        assert pu.FIT_PROTOCOL["tolerance_change"] == 1e-12
        assert pu.FIT_PROTOCOL["line_search_fn"] == "strong_wolfe"
        assert pu.L2_LAMBDA == 1e-3
        assert pu.FIT_PROTOCOL["intercept_penalty"] == "none"

    def test_strong_wolfe_is_actually_available_here(self):
        import inspect

        import torch

        assert "strong_wolfe" in inspect.getsource(torch.optim.LBFGS)

    def test_targets_place_a_draw_exactly_halfway(self):
        assert pu.OUTCOME_TARGETS == {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}

    def test_game_logit_is_red_minus_blue_plus_intercept(self):
        assert pu.game_logit(0.1, 0.5, 0.2) == pytest.approx(0.4)


class TestObjective:
    def test_matches_a_direct_bce_computation(self):
        logits = np.array([0.3, -1.2, 0.0])
        targets = np.array([1.0, 0.0, 0.5])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        expected = -np.mean(
            targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)
        )
        assert pu.objective_value(logits, targets, []) == pytest.approx(expected)

    def test_penalty_is_lambda_times_sum_of_squares(self):
        value = pu.objective_value([0.0], [0.5], [3.0, 4.0])
        baseline = pu.objective_value([0.0], [0.5], [])
        assert value - baseline == pytest.approx(1e-3 * 25.0)

    def test_large_logits_do_not_overflow(self):
        assert np.isfinite(pu.objective_value([800.0, -800.0], [1.0, 0.0], []))

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(pu.Phase10UtilityError):
            pu.objective_value([0.0, 1.0], [0.5], [])


class TestCollinearityDiagnostic:
    def test_rank_deficiency_is_the_recorded_one(self, scaler):
        rank = pu.train_feature_rank(scaler)
        assert rank["columns"] == 47
        assert rank["rank"] == 31
        assert rank["rank_deficiency"] == 16
