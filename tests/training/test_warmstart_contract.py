"""Phase 8 Agent 1: the frozen warm-start training contract.

These tests pin every learning-design decision Agent 1 froze: roster and
weights, schedule, setup sources, example schema, targets, baselines, loss
normalization, pilot matrix, acceptance thresholds, and held-out sealing.
Agents 2-7 rely on these values verbatim; any legitimate change is a new
reviewed contract version, and this file is where the old one fails loudly.
"""

import json
from pathlib import Path

import pytest

from stratego.engine.constants import TRAINING_RULES
from stratego.evaluation.registry import ALL_POLICY_IDS
from stratego.training import warmstart_contract as wc
from stratego.training import warmstart_seed as ws
from stratego.training.setup_source import LibrarySetupSource, SetupSourceError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"


class TestFrozenVersions:
    def test_the_five_agent_1_versions_are_frozen(self):
        assert wc.WARMSTART_TRAINING_CONTRACT_VERSION == "warmstart_training_contract_v1"
        assert ws.SYNTHETIC_CORPUS_VERSION == "synthetic_warmstart_corpus_v1"
        assert ws.DECISION_SAMPLER_VERSION == "warmstart_decision_sampler_v1"
        assert wc.WARMSTART_EXAMPLE_VERSION == "warmstart_example_v1"
        assert wc.WARMSTART_EVAL_VERSION == "warmstart_eval_v1"

    def test_the_live_upstream_stack_matches_the_frozen_expectation(self):
        assert wc.verify_frozen_upstream(include_library_digest=True) == []

    def test_the_contract_document_serializes_and_digests_stably(self):
        document = wc.contract_document()
        text = json.dumps(document, sort_keys=True)
        assert "warmstart_training_contract_v1" in text
        assert wc.contract_digest() == wc.contract_digest()
        for version in (
            wc.WARMSTART_TRAINING_CONTRACT_VERSION,
            ws.SYNTHETIC_CORPUS_VERSION,
            ws.DECISION_SAMPLER_VERSION,
            wc.WARMSTART_EXAMPLE_VERSION,
            wc.WARMSTART_EVAL_VERSION,
        ):
            assert version in text


class TestTeacherPopulation:
    def test_the_live_registry_matches_the_frozen_roster(self):
        assert wc.verify_teacher_roster() == []

    def test_exactly_ten_teachers_in_the_frozen_order(self):
        assert len(wc.EXPECTED_TEACHER_ROSTER) == wc.EXPECTED_TEACHER_COUNT == 10
        assert tuple(policy_id for policy_id, _, _ in wc.EXPECTED_TEACHER_ROSTER) == ALL_POLICY_IDS

    def test_the_frozen_policy_supervision_weights(self):
        assert wc.POLICY_SUPERVISION_WEIGHTS == {
            "strategic_rule_based": 1.0,
            "tactical_rule_based": 1.0,
            "basic_heuristic": 0.5,
            "random_legal": 0.0,
            "stress_scout_rush": 0.0,
            "stress_miner_rush": 0.0,
            "stress_draw_seeker": 0.0,
            "stress_berserker": 0.0,
            "stress_information_miser": 0.0,
            "stress_chaos": 0.0,
        }

    def test_policy_weight_lookup_rejects_unknown_teachers(self):
        assert wc.policy_weight("basic_heuristic") == 0.5
        with pytest.raises(wc.WarmstartContractError):
            wc.policy_weight("gpt_5")

    def test_zero_weight_teachers_still_supervise_value_and_belief(self):
        for row in wc.teacher_population():
            assert row["value_supervision"] is True
            assert row["belief_supervision"] is True

    def test_population_rows_carry_identity_and_behavior_contracts(self):
        rows = wc.teacher_population()
        assert len(rows) == 10
        for row in rows:
            assert row["implementation_path"] in (
                "stratego/evaluation/baselines.py",
                "stratego/evaluation/stress.py",
            )
            assert row["stochastic"] is True
            behavior = row["behavior_contract"]
            assert "derive_decision_seed" in behavior["decision_rng"]
            assert behavior["ranking_tie_break"] == (
                "descending score, then ascending action id"
            )
        margins = {row["policy_id"]: row["behavior_contract"]["selection_margin"] for row in rows}
        assert margins["random_legal"] is None
        assert margins["basic_heuristic"] == 0.75
        assert margins["tactical_rule_based"] == 0.5
        assert margins["strategic_rule_based"] == 0.5


class TestMatchupSchedule:
    def test_one_hundred_ordered_cells_in_red_major_order(self):
        cells = wc.ordered_matchup_cells()
        assert len(cells) == wc.EXPECTED_CELL_COUNT == 100
        tokens = wc.teacher_tokens()
        for cell in cells:
            assert cell["cell_index"] == cell["red_index"] * 10 + cell["blue_index"]
            assert cell["red_token"] == tokens[cell["red_index"]]
            assert cell["blue_token"] == tokens[cell["blue_index"]]
        assert [cell["cell_index"] for cell in cells] == list(range(100))

    def test_the_exact_game_schedule_is_frozen(self):
        schedule = wc.matchup_schedule()
        assert schedule["games_per_cell"] == {"train": 200, "validation": 40, "test": 40}
        assert schedule["totals"] == {
            "train": 20000,
            "validation": 4000,
            "test": 4000,
            "total": 28000,
        }

    def test_game_identities_enumerate_the_full_schedule_uniquely(self):
        seen = set()
        for split, expected in (("train", 20000), ("validation", 4000), ("test", 4000)):
            identities = list(wc.iter_game_identities(split))
            assert len(identities) == expected
            seen.update(identity[-1] for identity in identities)
        assert len(seen) == 28000

    def test_unknown_splits_are_rejected(self):
        with pytest.raises(wc.WarmstartContractError):
            list(wc.iter_game_identities("holdout"))


class TestSetupSources:
    def test_the_train_source_is_the_production_entry_point(self):
        source = wc.corpus_setup_source("train")
        assert isinstance(source, LibrarySetupSource)
        assert source.split == "train"
        assert source.purpose == "training"
        assert source.profile == "neutral_v1"

    def test_held_out_sources_carry_the_frozen_justifications(self):
        validation = wc.corpus_setup_source("validation")
        assert validation.split == "validation"
        assert validation.purpose == "evaluation_audit"
        assert validation.access_justification == (
            "Phase 8 held-out warm-start validation corpus"
        )
        test = wc.corpus_setup_source("test")
        assert test.split == "test"
        assert test.access_justification == (
            "Phase 8 sealed held-out warm-start test corpus"
        )

    def test_a_training_purpose_source_cannot_reach_a_held_out_split(self):
        with pytest.raises(SetupSourceError):
            LibrarySetupSource(split="test", purpose="training")

    def test_the_frozen_assign_constants(self):
        assert wc.SETUP_SOURCE_ENVIRONMENT_ID == 0
        assert wc.SETUP_SOURCE_GENERATION == 0
        configuration = wc.setup_source_configuration()
        for split in ("train", "validation", "test"):
            call = configuration["per_split"][split]["assign_call"]
            assert call["environment_id"] == 0
            assert call["generation"] == 0
            assert call["root_seed"] == "warmstart_seed.setup_root_seed(game_id)"

    def test_assignment_is_deterministic_and_sides_are_independent(self):
        source = wc.corpus_setup_source("train")
        game_id = ws.synthetic_game_id(
            "train", "strategic_rule_based@1.1.0", "random_legal@1.0.0", 0
        )
        root_seed = ws.setup_root_seed(game_id)
        first = source.assign(
            root_seed=root_seed,
            environment_id=wc.SETUP_SOURCE_ENVIRONMENT_ID,
            generation=wc.SETUP_SOURCE_GENERATION,
            game_id=game_id,
        )
        second = source.assign(
            root_seed=root_seed,
            environment_id=wc.SETUP_SOURCE_ENVIRONMENT_ID,
            generation=wc.SETUP_SOURCE_GENERATION,
            game_id=game_id,
        )
        assert first.red_setup == second.red_setup
        assert first.blue_setup == second.blue_setup
        provenance = first.provenance
        assert provenance["split"] == "train"
        assert provenance["red"]["side_seed"] != provenance["blue"]["side_seed"]

    def test_corpus_games_use_the_frozen_training_rules(self):
        assert wc.CORPUS_RULES is TRAINING_RULES
        assert wc.CORPUS_RULES.battleless_move_limit == 100
        assert wc.CORPUS_RULES.absolute_move_limit == 4000


class TestExampleSchema:
    def test_the_exact_field_names_in_order(self):
        assert [field["name"] for field in wc.EXAMPLE_FIELDS] == [
            "observation",
            "legal_mask",
            "acting_player",
            "policy_action_abs",
            "policy_action_model",
            "policy_weight",
            "value_target",
            "belief_target",
            "belief_mask",
            "game_id",
            "decision_index",
            "source_policy_id",
            "corpus_split",
        ]

    def test_only_the_observation_is_a_model_input(self):
        model_inputs = [field["name"] for field in wc.EXAMPLE_FIELDS if field["model_input"]]
        assert model_inputs == ["observation"]
        schema = wc.example_schema()
        assert schema["model_input_fields"] == ["observation"]
        assert schema["example_version"] == "warmstart_example_v1"

    def test_tensor_shapes_match_the_frozen_model_contract(self):
        by_name = {field["name"]: field for field in wc.EXAMPLE_FIELDS}
        assert by_name["observation"]["shape"] == (127, 10, 10)
        assert by_name["legal_mask"]["shape"] == (10000,)
        assert by_name["belief_target"]["shape"] == (100,)
        assert by_name["belief_mask"]["shape"] == (100,)
        assert by_name["observation"]["dtype"] == "float32"


class TestTargetSemantics:
    def test_value_semantics_follow_the_model_contract(self):
        targets = wc.target_semantics()
        assert targets["value"]["classes"] == ["WIN", "DRAW", "LOSS"]
        assert targets["value"]["indices"] == {"WIN": 0, "DRAW": 1, "LOSS": 2}
        assert targets["value"]["perspective"] == "acting player"
        assert targets["value"]["bootstrapping"].startswith("none")

    def test_policy_semantics_freeze_the_frame_conversion(self):
        policy = wc.target_semantics()["policy"]
        assert policy["frame"] == "perspective_normalized_squares"
        assert policy["engine_frame"] == "absolute_engine_squares"
        assert "absolute_action_to_model" in policy["conversion"]
        assert "policy_weight == 0" in policy["zero_weight_rule"]

    def test_belief_semantics_reuse_the_frozen_dense_target(self):
        belief = wc.target_semantics()["belief"]
        assert belief["belief_target_version"] == "dense_belief_target_v1"
        assert belief["square_frame"] == "perspective_normalized_squares"
        assert belief["type_count"] == 12
        assert belief["ignore_index"] == -100


class TestEvaluationContract:
    def test_epsilon_and_bootstrap_are_frozen(self):
        contract = wc.evaluation_contract()
        assert contract["log_epsilon"] == 1e-12
        assert contract["bootstrap"]["replicates"] == 10000
        assert contract["bootstrap"]["confidence"] == 0.95
        assert contract["bootstrap"]["unit"].startswith("game")
        assert contract["bootstrap"]["seeds"] == {
            "validation": 2026081306,
            "test": 2026081307,
        }

    def test_baseline_definitions_are_the_common_contract_ones(self):
        contract = wc.evaluation_contract()
        assert "1 / legal_count" in contract["policy_baseline"]["definition"]
        assert "train selected examples only" in contract["value_baseline"]["definition"]
        assert "unresolved_remaining_count" in contract["belief_baseline"]["definition"]
        assert contract["belief_baseline"]["top1_tie_break"] == (
            "lowest piece-type index among tied maxima"
        )

    def test_loss_normalization_is_per_component(self):
        loss = wc.loss_semantics()
        assert "sum_i(weight_i * CE_i) / sum_i(weight_i)" in loss["policy"]
        assert loss["value"].startswith("L_value = mean CE")
        assert "max(supervised square count, 1)" in loss["belief"]


class TestPilotMatrix:
    def test_exactly_six_predeclared_candidates(self):
        matrix = wc.pilot_matrix()
        assert len(matrix["candidates"]) == 6
        assert matrix["candidate_limit"] == 6
        identifiers = [candidate["candidate_id"] for candidate in matrix["candidates"]]
        assert identifiers == [
            "ws_pilot_lr1e-3_balanced",
            "ws_pilot_lr1e-3_policy_led",
            "ws_pilot_lr3e-4_balanced",
            "ws_pilot_lr3e-4_policy_led",
            "ws_pilot_lr1e-4_balanced",
            "ws_pilot_lr1e-4_policy_led",
        ]

    def test_only_the_allowed_dimensions_vary(self):
        candidates = wc.pilot_matrix()["candidates"]
        assert {candidate["learning_rate"] for candidate in candidates} == {1e-3, 3e-4, 1e-4}
        assert {candidate["loss_profile"] for candidate in candidates} == {
            "balanced",
            "policy_led",
        }
        for candidate in candidates:
            assert set(candidate) == {
                "candidate_id",
                "learning_rate",
                "loss_profile",
                "lambda_policy",
                "lambda_value",
                "lambda_belief",
            }

    def test_the_frozen_loss_profiles(self):
        assert wc.PILOT_LOSS_PROFILES == {
            "balanced": {"lambda_policy": 1.0, "lambda_value": 1.0, "lambda_belief": 1.0},
            "policy_led": {"lambda_policy": 1.0, "lambda_value": 0.5, "lambda_belief": 0.5},
        }

    def test_the_fixed_controls_hold_the_common_contract_values(self):
        controls = wc.PILOT_FIXED_CONTROLS
        assert controls["model"] == "C1"
        assert controls["precision"] == "float32"
        assert controls["batch_size"] == 256
        assert controls["optimizer"] == "AdamW"
        assert controls["gradient_clip_norm"] == 1.0
        assert controls["model_init_seed"] == 2026081302
        assert controls["update_budget"] == 5000
        assert controls["validation_cadence_updates"] == 500

    def test_selection_score_and_vetoes_are_frozen(self):
        selection = wc.PILOT_SELECTION
        assert "mean(r_policy, r_value, r_belief)" in selection["score"]
        assert "any component ratio > 1.05 at the final pilot checkpoint" in (
            selection["hard_veto"]
        )
        assert selection["tie_break_order"] == [
            "lower selection score",
            "lower validation policy ratio",
            "higher measured training examples/s",
        ]
        assert "test metrics" in selection["forbidden_evidence"]
        assert "Phase 4 game strength" in selection["forbidden_evidence"]

    def test_the_development_budget_caps(self):
        assert wc.DEVELOPMENT_BUDGET == {
            "pilot_candidates_max": 6,
            "pilot_updates_per_config_max": 5000,
            "final_run_optimizer_steps_max": 25000,
        }


class TestAcceptanceThresholds:
    def test_the_random_gate_is_verbatim(self):
        gate = wc.acceptance_thresholds()["playing_strength_vs_random"]
        assert gate["games"] == 2048
        assert gate["evaluation_pairs"] == 1024
        assert gate["effective_win_rate_min"] == 0.950
        assert gate["red_effective_win_rate_min"] == 0.900
        assert gate["blue_effective_win_rate_min"] == 0.900
        assert gate["paired_bootstrap_lower_bound_exclusive"] == 0.900
        assert gate["illegal_moves_max"] == 0
        assert gate["model_failures_max"] == 0
        assert gate["non_finite_outputs_max"] == 0
        assert gate["setup_bank_digest"] == (
            "5fe5f98750ca2bd90ee75a74b3ba024bf753342872ae5472f13eb7afbb674266"
        )

    def test_the_initialization_gate_is_verbatim(self):
        gate = wc.acceptance_thresholds()["improvement_over_initialization"]
        assert gate["paired_setup_cases_min"] == 512
        assert gate["games_min"] == 1024
        assert gate["effective_win_rate_min"] == 0.700
        assert gate["paired_bootstrap_lower_bound_exclusive"] == 0.550
        assert "seed=2026081302" in gate["opponent"]

    def test_the_learning_gates_are_verbatim(self):
        thresholds = wc.acceptance_thresholds()
        assert thresholds["policy_learning"]["ce_ratio_vs_uniform_legal_max"] == 0.90
        assert thresholds["policy_learning"]["top1_must_beat_uniform_expected_top1"] is True
        assert thresholds["value_learning"]["ce_ratio_vs_train_prior_max"] == 0.98
        assert thresholds["value_learning"]["brier_must_beat_train_prior"] is True
        assert thresholds["belief_learning"]["ce_ratio_vs_remaining_count_prior_max"] == 0.98
        assert thresholds["belief_learning"]["top1_must_beat_remaining_count_prior"] is True

    def test_the_stability_gate_is_verbatim(self):
        gate = wc.acceptance_thresholds()["stability"]
        assert gate["finite_logits_fraction_required"] == 1.0
        assert gate["max_legal_probability_threshold"] == 0.999
        assert gate["fraction_above_threshold_max_exclusive"] == 0.95


class TestHeldOutSealing:
    def test_structural_audit_is_always_allowed(self):
        for agent in range(1, 8):
            access = wc.check_test_corpus_access("structural_audit", phase8_agent=agent)
            assert access.resource == "test_corpus"

    def test_model_facing_test_purposes_raise_before_agent_7(self):
        for agent in range(1, 7):
            for purpose in (
                "model_inference",
                "model_metric",
                "checkpoint_selection",
                "hyperparameter_selection",
                "early_stopping",
                "final_evaluation",
            ):
                with pytest.raises(wc.HeldOutAccessError):
                    wc.check_test_corpus_access(purpose, phase8_agent=agent)

    def test_agent_7_opens_the_final_evaluation(self):
        access = wc.check_test_corpus_access("final_evaluation", phase8_agent=7)
        assert access.phase8_agent == 7
        with pytest.raises(wc.HeldOutAccessError):
            wc.check_test_corpus_access("model_inference", phase8_agent=7)

    def test_bank_regressions_stay_open_and_strength_stays_sealed(self):
        for agent in range(1, 8):
            wc.check_phase4_bank_access("non_neural_regression", phase8_agent=agent)
        for agent in range(1, 7):
            for purpose in (
                "neural_playing_strength",
                "pilot_selection",
                "config_selection",
                "checkpoint_selection",
                "final_random_evaluation",
            ):
                with pytest.raises(wc.HeldOutAccessError):
                    wc.check_phase4_bank_access(purpose, phase8_agent=agent)
        wc.check_phase4_bank_access("final_random_evaluation", phase8_agent=7)
        wc.check_phase4_bank_access("final_ladder_evaluation", phase8_agent=7)

    def test_unknown_purposes_and_agents_are_rejected(self):
        with pytest.raises(wc.HeldOutAccessError):
            wc.check_test_corpus_access("vibes", phase8_agent=3)
        with pytest.raises(wc.HeldOutAccessError):
            wc.check_phase4_bank_access("vibes", phase8_agent=3)
        with pytest.raises(wc.HeldOutAccessError):
            wc.check_test_corpus_access("structural_audit", phase8_agent=0)
        with pytest.raises(wc.HeldOutAccessError):
            wc.check_phase4_bank_access("non_neural_regression", phase8_agent=8)


class TestAgent1Artifacts:
    """PASS-gated once the Agent 1 runner has written its artifacts."""

    @pytest.fixture()
    def contract_artifact(self):
        path = ARTIFACT_DIRECTORY / "agent_01_warmstart_contract.json"
        if not path.exists():
            pytest.skip("Agent 1 contract artifact not written yet")
        return json.loads(path.read_text())

    @pytest.fixture()
    def teacher_artifact(self):
        path = ARTIFACT_DIRECTORY / "agent_01_teacher_population.json"
        if not path.exists():
            pytest.skip("Agent 1 teacher artifact not written yet")
        return json.loads(path.read_text())

    @pytest.fixture()
    def thresholds_artifact(self):
        path = ARTIFACT_DIRECTORY / "agent_01_acceptance_thresholds.json"
        if not path.exists():
            pytest.skip("Agent 1 thresholds artifact not written yet")
        return json.loads(path.read_text())

    def test_the_artifact_declares_pass_and_names_the_live_contract(self, contract_artifact):
        assert contract_artifact["status"] == "PASS"
        assert contract_artifact["contract"]["contract_version"] == (
            wc.WARMSTART_TRAINING_CONTRACT_VERSION
        )
        assert contract_artifact["contract_digest"] == wc.contract_digest()
        gates = contract_artifact["completion_gates"]
        assert gates and all(gates.values())

    def test_the_recorded_teachers_are_the_live_frozen_population(self, teacher_artifact):
        assert teacher_artifact["status"] == "PASS"
        assert teacher_artifact["teacher_population"] == teacher_population_as_json()
        assert teacher_artifact["policy_supervision_weights"] == {
            key: value for key, value in wc.POLICY_SUPERVISION_WEIGHTS.items()
        }

    def test_the_recorded_thresholds_are_the_live_frozen_thresholds(self, thresholds_artifact):
        assert thresholds_artifact["status"] == "PASS"
        assert thresholds_artifact["acceptance_thresholds"] == json.loads(
            json.dumps(wc.acceptance_thresholds())
        )
        assert thresholds_artifact["sealing_rules"] == json.loads(
            json.dumps(wc.sealing_rules())
        )


def teacher_population_as_json():
    """The live population after one JSON round trip (tuples become lists)."""
    return json.loads(json.dumps(wc.teacher_population()))
