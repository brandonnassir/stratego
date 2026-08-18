"""Unit coverage: Agent 3's fit path, its allowlist guard, and the audit path.

Everything here runs on a small synthetic corpus built from real train-split
library bases — no sealed corpus, no external volume, no game playing — so
the properties that make the fit trustworthy are checked in seconds:

- the allowlist guard is a wall, not a convention: reading any stored field
  outside the model's frozen allowlist raises;
- targets are rebuilt from the W/D/L token, never from `red_score`;
- the design orientation is `+u(red) - u(blue)` and the audit re-derives it;
- the frozen all-zero start is handled correctly (every logit starts at
  exactly 0.0, where a careless stable-BCE composition has a wrong autograd
  subgradient — the fit must actually descend, and its gradient at zero must
  equal the analytic `mean(sigmoid(0) - y)`);
- two fits are bit-identical; the audit agrees with the reported objective;
- the negative controls fire on corrupted inputs and stay quiet on clean
  ones; the exported artifact is a pure own-side scorer.
"""

import json

import numpy as np
import pytest

from stratego.setups.sampler import load_library_index
from stratego.training import phase10_utility_audit as audit
from stratego.training import phase10_utility_fit as fit
from stratego.training.phase10_outcome_store import ASSEMBLED_RECORD_FIELDS
from stratego.training.phase10_seed import phase10_game_id
from stratego.training.phase10_utility import (
    TRAIT_FEATURE_NAMES,
    fit_trait_scaler,
    trait_feature_vector,
)


@pytest.fixture(scope="module")
def index():
    return load_library_index()


@pytest.fixture(scope="module")
def scaler():
    return fit_trait_scaler()


def _record(red_family, blue_family, ordinal, result, red_base, blue_base):
    return {
        "game_id": phase10_game_id(red_family, blue_family, ordinal),
        "red_family": red_family,
        "blue_family": blue_family,
        "result": result,
        "red_base_setup_id": red_base.base_setup_id,
        "blue_base_setup_id": blue_base.base_setup_id,
        # Forbidden-by-allowlist fields, present to prove they are not read.
        "red_score": {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}[result],
        "plies": 100 + ordinal,
        "terminal_reason": "flag_captured",
        "red_final_fingerprint": "not-a-feature",
        "match_seed": 12345,
    }


@pytest.fixture(scope="module")
def records(index):
    """A non-degenerate synthetic corpus: asymmetric ordered family pairs."""
    f00 = index.eligible_bases("F00", "train")[:4]
    f01 = index.eligible_bases("F01", "train")[:4]
    built = []
    # F00 as Red beats F01 (mostly); F01 as Red loses to F00 (mostly), with
    # draws sprinkled in so the frozen 0.5 handling is exercised.
    for ordinal in range(32):
        result = "draw" if ordinal % 8 == 7 else ("red_win" if ordinal % 4 else "red_loss")
        built.append(_record("F00", "F01", ordinal, result, f00[ordinal % 4], f01[(ordinal // 4) % 4]))
    for ordinal in range(32):
        result = "draw" if ordinal % 8 == 3 else ("red_loss" if ordinal % 4 else "red_win")
        built.append(_record("F01", "F00", ordinal, result, f01[ordinal % 4], f00[(ordinal // 4) % 4]))
    return sorted(built, key=lambda record: record["game_id"])


@pytest.fixture(scope="module")
def fitted_models(records, index, scaler):
    models = {}
    for model_id in ("model_F", "model_T"):
        data = fit.build_fit_data(records, model_id, index=index, scaler=scaler)
        models[model_id] = (data, fit.fit_utility_model(data))
    return models


@pytest.fixture(scope="module")
def design(fitted_models):
    data, _ = fitted_models["model_T"]
    return audit.AuditDesign(
        game_ids=data.game_ids,
        targets=data.targets,
        red_family_index=data.red_family_index,
        blue_family_index=data.blue_family_index,
        red_features=data.red_features,
        blue_features=data.blue_features,
    )


class TestAllowlistGuard:
    def test_forbidden_field_access_raises(self, records):
        view = fit.AllowlistedRecord(records[0], fit.FIT_INPUT_ALLOWLIST["model_F"])
        for forbidden in ("plies", "red_score", "terminal_reason", "red_final_fingerprint", "match_seed"):
            with pytest.raises(fit.Phase10UtilityFitError):
                view[forbidden]

    def test_model_f_cannot_read_base_ids(self, records):
        view = fit.AllowlistedRecord(records[0], fit.FIT_INPUT_ALLOWLIST["model_F"])
        with pytest.raises(fit.Phase10UtilityFitError):
            view["red_base_setup_id"]

    def test_accessed_fields_are_recorded(self, records):
        view = fit.AllowlistedRecord(records[0], fit.FIT_INPUT_ALLOWLIST["model_F"])
        view["game_id"], view["result"]
        assert view.accessed == {"game_id", "result"}

    def test_the_forbidden_complement_is_total(self):
        widest = set(fit.FIT_INPUT_ALLOWLIST["model_T"])
        assert widest | set(fit.FORBIDDEN_FIT_FIELDS) >= set(ASSEMBLED_RECORD_FIELDS)
        assert not widest & set(fit.FORBIDDEN_FIT_FIELDS)
        for field in (
            "red_provenance",
            "blue_setup_draw_seed",
            "red_setup_attempt",
            "decisions",
            "winner",
            "payload_digest",
            "commit_digest",
            "move_policy_identity",
        ):
            assert field in fit.FORBIDDEN_FIT_FIELDS

    def test_fit_data_reads_only_the_allowlist(self, records, index, scaler):
        data = fit.build_fit_data(records, "model_T", index=index, scaler=scaler)
        assert set(data.accessed_fields) <= set(fit.FIT_INPUT_ALLOWLIST["model_T"])
        data_f = fit.build_fit_data(records, "model_F", index=index, scaler=scaler)
        assert set(data_f.accessed_fields) <= set(fit.FIT_INPUT_ALLOWLIST["model_F"])
        assert "red_base_setup_id" not in data_f.accessed_fields


class TestDesignBuilding:
    def test_targets_come_from_the_result_token(self, records, index, scaler):
        data = fit.build_fit_data(records, "model_F", index=index, scaler=scaler)
        expected = np.array(
            [{"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}[record["result"]] for record in records]
        )
        assert np.array_equal(data.targets, expected)
        assert set(np.unique(data.targets)) <= {0.0, 0.5, 1.0}

    def test_orientation_red_then_blue(self, records, index, scaler):
        data = fit.build_fit_data(records, "model_T", index=index, scaler=scaler)
        first = records[0]
        assert data.red_family_index[0] == int(first["red_family"][1:])
        assert data.blue_family_index[0] == int(first["blue_family"][1:])
        red_entry = index.base(first["red_base_setup_id"])
        expected = scaler.transform(trait_feature_vector(red_entry.trait_vector))
        assert np.array_equal(data.red_features[0], expected)

    def test_out_of_order_records_are_rejected(self, records, index, scaler):
        shuffled = [records[1], records[0], *records[2:]]
        with pytest.raises(fit.Phase10UtilityFitError, match="canonical"):
            fit.build_fit_data(shuffled, "model_F", index=index, scaler=scaler)

    def test_unknown_result_is_rejected(self, records, index, scaler):
        corrupted = [dict(records[0], result="red_forfeit"), *records[1:]]
        corrupted.sort(key=lambda record: record["game_id"])
        with pytest.raises(fit.Phase10UtilityFitError, match="result"):
            fit.build_fit_data(corrupted, "model_F", index=index, scaler=scaler)

    def test_held_out_base_is_a_hard_stop(self, records, index, scaler):
        held_out = index.eligible_bases("F00", "validation")[0]
        corrupted = [dict(records[0]), *records[1:]]
        corrupted[0]["red_base_setup_id"] = held_out.base_setup_id
        with pytest.raises(fit.Phase10UtilityFitError, match="held-out|split"):
            fit.build_fit_data(corrupted, "model_T", index=index, scaler=scaler)

    def test_family_mismatch_is_a_hard_stop(self, records, index, scaler):
        wrong_family = index.eligible_bases("F05", "train")[0]
        corrupted = [dict(records[0]), *records[1:]]
        corrupted[0]["red_base_setup_id"] = wrong_family.base_setup_id
        with pytest.raises(fit.Phase10UtilityFitError, match="family"):
            fit.build_fit_data(corrupted, "model_T", index=index, scaler=scaler)

    def test_wrong_scaler_is_refused(self, records, index, scaler):
        from stratego.training.phase10_utility import TraitScaler

        wrong = TraitScaler(scaler.mean + 1.0, scaler.std, base_count=6400)
        with pytest.raises(fit.Phase10UtilityFitError, match="scaler"):
            fit.build_fit_data(records, "model_T", index=index, scaler=wrong)


class TestFrozenFit:
    def test_gradient_at_the_all_zero_start_is_analytic(self, fitted_models):
        """At zero parameters every logit is exactly 0; dL/dc must be mean(0.5 - y)."""
        data, _ = fitted_models["model_F"]
        zero = {
            "model_id": "model_F",
            "red_first_intercept": 0.0,
            "family_offsets_raw": [[0.0] * 16, [0.0] * 16],
            "family_offsets_effective": [[0.0] * 16, [0.0] * 16],
            "trait_weights": None,
            "colour_order": ["red", "blue"],
            "family_order": [f"F{i:02d}" for i in range(16)],
            "feature_order": [],
            "diagnostics": {"objective": 0.0, "bce": 0.0, "l2_penalty": 0.0},
        }
        design = audit.AuditDesign(
            game_ids=data.game_ids,
            targets=data.targets,
            red_family_index=data.red_family_index,
            blue_family_index=data.blue_family_index,
            red_features=np.zeros((len(data.game_ids), 47)),
            blue_features=np.zeros((len(data.game_ids), 47)),
        )
        gradient = audit.audit_gradient(design, zero)
        expected_intercept = float(np.mean(0.5 - data.targets))
        assert gradient["flat"][0] == pytest.approx(expected_intercept, abs=1e-15)

    def test_the_fit_actually_descends_from_zero(self, fitted_models):
        for model_id in ("model_F", "model_T"):
            _, fitted = fitted_models[model_id]
            assert fitted.diagnostics["objective"] < 0.6931471805599453
            assert fitted.diagnostics["iterations"] > 1

    def test_two_fits_are_bit_identical(self, fitted_models):
        for model_id in ("model_F", "model_T"):
            data, fitted = fitted_models[model_id]
            again = fit.fit_utility_model(data)
            assert fitted.coefficient_document() == again.coefficient_document()
            assert fitted.coefficient_digest() == again.coefficient_digest()
            assert fitted.diagnostics["objective"] == again.diagnostics["objective"]

    def test_raw_offsets_self_center(self, fitted_models):
        for model_id in ("model_F", "model_T"):
            _, fitted = fitted_models[model_id]
            for row in fitted.family_offsets_raw:
                assert abs(sum(row) / len(row)) < 1e-8
            for row in fitted.family_offsets_effective:
                assert abs(sum(row) / len(row)) < 1e-14

    def test_parameter_shapes_match_the_frozen_layout(self, fitted_models):
        _, fitted_f = fitted_models["model_F"]
        assert len(fitted_f.family_offsets_raw) == 2
        assert all(len(row) == 16 for row in fitted_f.family_offsets_raw)
        assert fitted_f.trait_weights is None
        _, fitted_t = fitted_models["model_T"]
        assert len(fitted_t.trait_weights) == 2
        assert all(len(row) == 47 for row in fitted_t.trait_weights)

    def test_model_f_data_refuses_model_t_features(self, fitted_models):
        data_t, _ = fitted_models["model_T"]
        wrong = fit.FitData(
            model_id="model_F",
            game_ids=data_t.game_ids,
            targets=data_t.targets,
            red_family_index=data_t.red_family_index,
            blue_family_index=data_t.blue_family_index,
            red_features=data_t.red_features,
            blue_features=data_t.blue_features,
            scaler_digest=data_t.scaler_digest,
            accessed_fields=data_t.accessed_fields,
        )
        with pytest.raises(fit.Phase10UtilityFitError):
            fit.fit_utility_model(wrong)


class TestIndependentAudit:
    def test_audit_agrees_with_the_reported_objective(self, fitted_models, design):
        for model_id in ("model_F", "model_T"):
            _, fitted = fitted_models[model_id]
            outcome = audit.audit_fitted_model(design, fitted.to_dict())
            assert outcome["all_pass"], outcome["checks"]
            assert outcome["objective_abs_difference"] <= audit.OBJECTIVE_AGREEMENT_TOLERANCE

    def test_independent_flattening_matches_the_frozen_order(self, index):
        assert audit.independent_feature_names() == TRAIT_FEATURE_NAMES
        entry = index.eligible_bases("F03", "train")[0]
        assert audit.independent_flatten(entry.trait_vector) == trait_feature_vector(
            entry.trait_vector
        )

    def test_refit_comparator_detects_differences(self, fitted_models):
        _, fitted = fitted_models["model_F"]
        document = fitted.coefficient_document()
        assert audit.compare_refits([document, json.loads(json.dumps(document))])["identical"]
        tampered = json.loads(json.dumps(document))
        tampered["family_offsets_raw"][0][0] += 1e-12
        comparison = audit.compare_refits([document, tampered])
        assert not comparison["identical"]
        assert comparison["max_abs_difference"] > 0.0


class TestNegativeControls:
    def test_orientation_swap_fires(self, fitted_models, design):
        for model_id in ("model_F", "model_T"):
            _, fitted = fitted_models[model_id]
            control = audit.control_orientation_swap(design, fitted.to_dict())
            assert control["detected"], control

    def test_wrong_draw_target_fires(self, fitted_models, design, records):
        _, fitted = fitted_models["model_T"]
        control = audit.control_wrong_draw_target(design, fitted.to_dict(), records)
        assert control["detected"]
        assert control["draws_in_corpus"] > 0
        assert control["target_mismatches_detected"] == control["draws_in_corpus"]

    def test_permuted_trait_column_fires(self, fitted_models, design):
        _, fitted = fitted_models["model_T"]
        control = audit.control_permuted_trait_column(design, fitted.to_dict())
        assert control["detected"]

    def test_altered_coefficient_fires(self, fitted_models, design):
        for model_id in ("model_F", "model_T"):
            _, fitted = fitted_models[model_id]
            control = audit.control_altered_coefficient(design, fitted.to_dict())
            assert control["detected"]

    def test_controls_stay_quiet_on_clean_inputs(self, fitted_models, design):
        """Non-vacuity: the same checks pass on the true design."""
        _, fitted = fitted_models["model_T"]
        outcome = audit.audit_fitted_model(design, fitted.to_dict())
        assert outcome["all_pass"]


@pytest.fixture(scope="module")
def artifact(fitted_models, scaler):
    models = {model_id: fitted for model_id, (_, fitted) in fitted_models.items()}
    return fit.utility_models_artifact(
        models, scaler, corpus_content_digest="0" * 64, corpus_games=64
    )


class TestProductionScorer:
    def test_artifact_is_own_side_only(self, artifact):
        assert fit.own_side_only_findings(artifact) == []

    def test_a_smuggled_opponent_table_is_detected(self, artifact):
        tampered = json.loads(json.dumps(artifact))
        tampered["opponent_matchup_matrix"] = [[0.0] * 16] * 16
        findings = fit.own_side_only_findings(tampered)
        assert findings
        with pytest.raises(fit.Phase10UtilityFitError):
            fit.SetupUtilityScorer(tampered)

    def test_scoring_needs_only_own_side_inputs(self, artifact, index):
        scorer = fit.SetupUtilityScorer(artifact)
        entry = index.eligible_bases("F00", "train")[0]
        value_f = scorer.utility("model_F", "red", "F00")
        value_t = scorer.utility("model_T", "blue", "F00", entry.trait_vector)
        assert np.isfinite(value_f) and np.isfinite(value_t)

    def test_the_intercept_never_reaches_a_score(self, artifact, index):
        tampered = json.loads(json.dumps(artifact))
        for model_id in ("model_F", "model_T"):
            tampered["models"][model_id]["red_first_intercept"] = 999.0
        entry = index.eligible_bases("F00", "train")[0]
        original = fit.SetupUtilityScorer(artifact)
        shifted = fit.SetupUtilityScorer(tampered)
        assert original.utility("model_F", "red", "F00") == shifted.utility(
            "model_F", "red", "F00"
        )
        assert original.utility("model_T", "red", "F00", entry.trait_vector) == shifted.utility(
            "model_T", "red", "F00", entry.trait_vector
        )

    def test_scorer_decomposes_the_training_logit(self, artifact, fitted_models, design, records, index):
        scorer = fit.SetupUtilityScorer(artifact)
        for model_id in ("model_F", "model_T"):
            _, fitted = fitted_models[model_id]
            eta = audit.audit_logits(design, fitted.to_dict())
            for position in (0, 17, 63):
                record = records[position]
                red_trait = blue_trait = None
                if model_id == "model_T":
                    red_trait = index.base(record["red_base_setup_id"]).trait_vector
                    blue_trait = index.base(record["blue_base_setup_id"]).trait_vector
                decomposed = (
                    fitted.red_first_intercept
                    + scorer.utility(model_id, "red", record["red_family"], red_trait)
                    - scorer.utility(model_id, "blue", record["blue_family"], blue_trait)
                )
                assert decomposed == pytest.approx(float(eta[position]), abs=1e-10)
