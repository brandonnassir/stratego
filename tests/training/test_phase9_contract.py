"""Regression: the frozen Phase 9 RL contract stays exactly what Agent 1 froze.

Every constant, formula and access rule pinned here is a learning-design
decision the sequence document assigns to Agent 1 alone. Later agents
implement these values; a failing test means a frozen semantic drifted,
which requires a reviewed new contract version, never an in-place edit.
"""

import json
import math
from collections import Counter

import numpy as np
import pytest

from stratego.engine.constants import BLUE, RED
from stratego.training import phase9_contract as pc
from stratego.training import phase9_seed as ps


class TestContractDocument:
    def test_round_trip_through_json(self):
        document = pc.rl_contract_document()
        rebuilt = json.loads(json.dumps(document))
        assert rebuilt == json.loads(json.dumps(pc.rl_contract_document()))

    def test_contract_digest_is_stable(self):
        assert pc.contract_digest() == pc.contract_digest()

    def test_all_nine_contract_identities_are_frozen(self):
        assert pc.CONTRACT_IDENTITIES == (
            "phase9_rl_contract_v1",
            "phase9_population_v1",
            "phase9_rollout_schedule_v1",
            "phase9_rollout_store_v1",
            "phase9_advantage_v1",
            "phase9_train_order_v1",
            "phase9_checkpoint_v1",
            "phase9_eval_bank_v1",
            "phase9_acceptance_v1",
        )

    def test_frozen_phase8_inputs(self):
        inputs = pc.rl_contract_document()["frozen_phase8_inputs"]
        assert inputs["checkpoint_sha256"] == (
            "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
        )
        assert inputs["canonical_untrained_sha256"] == (
            "01c907eeef86ec04121db55ccffb9365e8df27fdf05921b921947d4af365754c"
        )
        assert inputs["selected_update"] == 24000
        assert inputs["c1_parameters"] == 863959
        assert inputs["canonical_init_seed"] == 2026081302
        corpus = inputs["corpus"]
        assert corpus["resolver"] == (
            "stratego.training.synthetic_corpus.default_corpus_root()"
        )
        assert corpus["content_digest"].startswith("c95c3545")
        assert corpus["metadata_digest"].startswith("1db0f02f")
        assert corpus["commit_index_digest"].startswith("32e8e18d")

    def test_live_upstream_matches_the_frozen_expectations(self):
        assert pc.verify_phase9_upstream(include_library_digest=False) == []


class TestPopulationArithmetic:
    def test_canonical_counts_are_exact(self):
        assert pc.CANONICAL_BUCKET_COUNTS == {
            "current": 1024,
            "historical": 512,
            "rule": 307,
            "stress": 205,
        }
        assert sum(pc.CANONICAL_BUCKET_COUNTS.values()) == 2048
        assert pc.CANONICAL_RULE_TIER_COUNTS == {
            "strategic_rule_based": 154,
            "tactical_rule_based": 107,
            "basic_heuristic": 46,
        }
        assert sum(pc.CANONICAL_RULE_TIER_COUNTS.values()) == 307

    def test_pilot_counts_are_exact(self):
        assert pc.PILOT_BUCKET_COUNTS == {
            "current": 512,
            "historical": 256,
            "rule": 154,
            "stress": 102,
        }
        assert sum(pc.PILOT_BUCKET_COUNTS.values()) == 1024
        assert pc.PILOT_RULE_TIER_COUNTS == {
            "strategic_rule_based": 77,
            "tactical_rule_based": 54,
            "basic_heuristic": 23,
        }
        assert sum(pc.PILOT_RULE_TIER_COUNTS.values()) == 154

    def test_counts_realize_the_frozen_proportions(self):
        for counts in (pc.CANONICAL_BUCKET_COUNTS, pc.PILOT_BUCKET_COUNTS):
            total = sum(counts.values())
            for bucket, proportion in pc.POPULATION_PROPORTIONS.items():
                assert abs(counts[bucket] / total - proportion) <= 1.0 / total

    def test_rule_subranges_are_contiguous_and_exact(self):
        observed = Counter(
            pc.rule_tier_for_ordinal("canonical", ordinal) for ordinal in range(307)
        )
        assert observed == Counter(pc.CANONICAL_RULE_TIER_COUNTS)
        assert pc.rule_tier_for_ordinal("canonical", 0) == "strategic_rule_based"
        assert pc.rule_tier_for_ordinal("canonical", 153) == "strategic_rule_based"
        assert pc.rule_tier_for_ordinal("canonical", 154) == "tactical_rule_based"
        assert pc.rule_tier_for_ordinal("canonical", 260) == "tactical_rule_based"
        assert pc.rule_tier_for_ordinal("canonical", 261) == "basic_heuristic"
        assert pc.rule_tier_for_ordinal("canonical", 306) == "basic_heuristic"
        observed_pilot = Counter(
            pc.rule_tier_for_ordinal("pilot_p9a", ordinal) for ordinal in range(154)
        )
        assert observed_pilot == Counter(pc.PILOT_RULE_TIER_COUNTS)
        with pytest.raises(pc.Phase9ContractError):
            pc.rule_tier_for_ordinal("canonical", 307)

    def test_stress_rotation_is_balanced(self):
        one_iteration = Counter(
            pc.stress_policy_for_ordinal(1, ordinal, namespace="canonical")
            for ordinal in range(205)
        )
        assert set(one_iteration) == set(pc.STRESS_POLICY_ROSTER)
        assert max(one_iteration.values()) - min(one_iteration.values()) == 1
        six_iterations = Counter(
            pc.stress_policy_for_ordinal(iteration, ordinal, namespace="canonical")
            for iteration in range(1, 7)
            for ordinal in range(205)
        )
        assert all(count == 205 for count in six_iterations.values())

    def test_scheduled_iteration_is_exact_and_unique(self):
        games = list(pc.iter_scheduled_games("canonical", 1))
        assert len(games) == 2048
        assert len({game["game_id"] for game in games}) == 2048
        buckets = Counter(game["bucket"] for game in games)
        assert buckets == Counter(pc.CANONICAL_BUCKET_COUNTS)
        pilot_games = list(pc.iter_scheduled_games("pilot_p9f", 8))
        assert len(pilot_games) == 1024
        assert Counter(game["bucket"] for game in pilot_games) == Counter(
            pc.PILOT_BUCKET_COUNTS
        )


class TestLearnerControlAndColorBalance:
    def test_self_play_is_both(self):
        assert pc.learner_control_for("current", 1, 0) == "both"
        assert pc.learner_color("current", 1, 0) is None

    def test_parity_rule(self):
        assert pc.learner_color("rule", 1, 1) == RED
        assert pc.learner_color("rule", 1, 0) == BLUE
        assert pc.learner_color("rule", 2, 0) == RED
        assert pc.learner_control_for("historical", 2, 0) == "red"
        assert pc.learner_control_for("historical", 2, 1) == "blue"

    def test_exact_split_and_remainder_alternation(self):
        for bucket, size in (("rule", 307), ("stress", 205), ("historical", 512)):
            reds_odd_iteration = sum(
                1 for ordinal in range(size) if pc.learner_color(bucket, 1, ordinal) == RED
            )
            reds_even_iteration = sum(
                1 for ordinal in range(size) if pc.learner_color(bucket, 2, ordinal) == RED
            )
            assert reds_odd_iteration + reds_even_iteration == size
            assert abs(reds_odd_iteration - reds_even_iteration) == size % 2

    def test_training_eligibility_table(self):
        assert pc.TRAINING_ELIGIBILITY == {
            "current": "both colors",
            "historical": "current-policy side only",
            "rule": "current-policy side only",
            "stress": "current-policy side only",
        }


class TestHistoricalLeague:
    def test_anchor_and_cadence(self):
        assert pc.HISTORICAL_ANCHOR_ID == "H000"
        assert pc.ARCHIVE_CADENCE_ITERATIONS == 5
        assert pc.ACTIVE_WINDOW_RECENT_SNAPSHOTS == 8

    def test_archive_identity_rule(self):
        assert pc.archive_snapshot_id(5) == "H005"
        assert pc.archive_snapshot_id(60) == "H060"
        with pytest.raises(pc.Phase9ContractError):
            pc.archive_snapshot_id(7)

    def test_active_window_growth_and_cap(self):
        assert pc.active_historical_window(1) == ("H000",)
        assert pc.active_historical_window(5) == ("H000",)
        assert pc.active_historical_window(6) == ("H000", "H005")
        assert pc.active_historical_window(41) == (
            "H000", "H005", "H010", "H015", "H020", "H025", "H030", "H035", "H040",
        )
        assert pc.active_historical_window(60) == (
            "H000", "H020", "H025", "H030", "H035", "H040", "H045", "H050", "H055",
        )
        assert len(pc.active_historical_window(60)) == 9

    def test_historical_draw_is_deterministic_and_in_window(self):
        for iteration in (1, 6, 23, 60):
            window = pc.active_historical_window(iteration)
            for ordinal in range(16):
                game_id = ps.phase9_game_id("canonical", iteration, "historical", ordinal)
                first = pc.historical_opponent_for(game_id)
                assert first in window
                assert first == pc.historical_opponent_for(game_id)

    def test_draws_cover_the_window(self):
        window = set(pc.active_historical_window(60))
        observed = {
            pc.historical_opponent_for(
                ps.phase9_game_id("canonical", 60, "historical", ordinal)
            )
            for ordinal in range(512)
        }
        assert observed == window


class TestAdvantageMathematics:
    def test_behavior_value_scalar(self):
        assert pc.behavior_value_scalar((0.6, 0.3, 0.1)) == pytest.approx(0.5)
        assert pc.behavior_value_scalar((1.0, 0.0, 0.0)) == 1.0

    def test_terminal_z(self):
        assert pc.terminal_z("win") == 1
        assert pc.terminal_z("draw") == 0
        assert pc.terminal_z("loss") == -1
        with pytest.raises(pc.Phase9ContractError):
            pc.terminal_z("timeout")

    def test_single_decision_sequence(self):
        assert pc.temporal_deltas([0.3], -1) == pytest.approx([-1.3])
        assert pc.advantages([0.3], -1) == pytest.approx([-1.3])

    def test_two_step_hand_computation(self):
        # v = [0.2, 0.5], win: deltas [0.3, 0.5]; A1 = 0.5, A0 = 0.3 + 0.5*0.5
        assert pc.temporal_deltas([0.2, 0.5], 1) == pytest.approx([0.3, 0.5])
        assert pc.advantages([0.2, 0.5], 1) == pytest.approx([0.55, 0.5])

    def test_three_step_hand_computation(self):
        # v = [0.1, -0.2, 0.4], draw: deltas [-0.3, 0.6, -0.4]
        # A2 = -0.4; A1 = 0.6 + 0.5*(-0.4) = 0.4; A0 = -0.3 + 0.5*0.4 = -0.1
        values = [0.1, -0.2, 0.4]
        assert pc.temporal_deltas(values, 0) == pytest.approx([-0.3, 0.6, -0.4])
        assert pc.advantages(values, 0) == pytest.approx([-0.1, 0.4, -0.4])

    def test_empty_sequence(self):
        assert pc.temporal_deltas([], 1) == []
        assert pc.advantages([], 1) == []

    def test_frozen_constants(self):
        assert pc.GAMMA == 1.0
        assert pc.LAMBDA_ADVANTAGE == 0.5
        assert pc.LAMBDA_VALUE == 0.8


class TestWdlLambdaTargets:
    def test_terminal_target_is_the_outcome(self):
        assert pc.wdl_lambda_targets([(0.5, 0.3, 0.2)], "loss") == [(0.0, 0.0, 1.0)]

    def test_two_step_hand_computation(self):
        targets = pc.wdl_lambda_targets([(0.6, 0.3, 0.1), (0.7, 0.2, 0.1)], "win")
        assert targets[1] == (1.0, 0.0, 0.0)
        assert targets[0] == pytest.approx((0.94, 0.04, 0.02))

    def test_three_step_hand_computation(self):
        predictions = [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.1, 0.2, 0.7)]
        targets = pc.wdl_lambda_targets(predictions, "loss")
        assert targets[2] == (0.0, 0.0, 1.0)
        # Y1 = 0.2*P2 + 0.8*Y2 = (0.02, 0.04, 0.94)
        assert targets[1] == pytest.approx((0.02, 0.04, 0.94))
        # Y0 = 0.2*P1 + 0.8*Y1 = (0.096, 0.112, 0.792)
        assert targets[0] == pytest.approx((0.096, 0.112, 0.792))

    def test_targets_stay_normalized(self):
        predictions = [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.1, 0.2, 0.7)]
        for outcome in ("win", "draw", "loss"):
            for target in pc.wdl_lambda_targets(predictions, outcome):
                assert sum(target) == pytest.approx(1.0)

    def test_unknown_outcome_is_refused(self):
        with pytest.raises(pc.Phase9ContractError):
            pc.wdl_lambda_targets([(0.5, 0.3, 0.2)], "timeout")


class TestAdvantageFilter:
    def test_quantile_matches_numpy_linear(self):
        values = sorted([0.004, 0.1, 0.2, 0.3, 0.55, 0.9, 1.4])
        for probability in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert pc.quantile_linear(values, probability) == pytest.approx(
                float(np.quantile(np.asarray(values), probability))
            )

    def test_hand_computed_threshold(self):
        # |A| sorted = [0.004, 0.1, 0.2, 0.3]; Q75 at index 2.25 -> 0.225
        assert pc.advantage_filter_threshold([0.1, -0.2, 0.3, 0.004]) == pytest.approx(0.225)

    def test_floor_applies(self):
        assert pc.advantage_filter_threshold([0.001, -0.002, 0.0005, 0.0]) == 0.01

    def test_empty_iteration_is_refused(self):
        with pytest.raises(pc.Phase9ContractError):
            pc.advantage_filter_threshold([])

    def test_frozen_filter_constants(self):
        assert pc.ADVANTAGE_FILTER_QUANTILE == 0.75
        assert pc.ADVANTAGE_FILTER_FLOOR == 0.01
        assert pc.ADVANTAGE_STANDARDIZATION_EPSILON == 1e-8
        semantics = pc.advantage_semantics()
        assert "PPO subset" in semantics["standardization"] or (
            "PPO-selected" in semantics["standardization"]
        )


class TestPpoAndDamping:
    def test_frozen_constants(self):
        assert pc.PPO_CLIP_EPSILON == 0.20
        assert pc.BEHAVIOR_KL_TARGET == 0.015
        assert pc.KL_HARD_LIMIT == 0.08
        assert pc.CLIP_FRACTION_HARD_LIMIT == 0.75
        assert pc.VALUE_LOSS_WEIGHT == 0.5
        assert pc.BELIEF_LOSS_WEIGHT == 0.25
        assert pc.MINIBATCH_SIZE == 512
        assert pc.EPOCHS_PER_ROLLOUT == 2

    def test_optimizer_constraints(self):
        constraints = pc.OPTIMIZER_CONSTRAINTS
        assert constraints["precision"] == "float32"
        assert constraints["device"] == "mps"
        assert constraints["optimizer"] == "AdamW"
        assert constraints["weight_decay"] == 0.01
        assert constraints["gradient_clip_norm"] == 1.0
        assert constraints["adam_betas"] == (0.9, 0.999)
        assert constraints["adam_epsilon"] == 1e-8

    def test_adaptive_beta_rule(self):
        assert pc.adaptive_kl_beta(0.01, 0.0301) == pytest.approx(0.02)
        assert pc.adaptive_kl_beta(0.01, 0.0074) == pytest.approx(0.005)
        assert pc.adaptive_kl_beta(0.01, 0.015) == pytest.approx(0.01)
        assert pc.adaptive_kl_beta(0.0300, 0.0300) == pytest.approx(0.0300)
        assert pc.adaptive_kl_beta(0.0075, 0.0075) == pytest.approx(0.0075)

    def test_beta_clamp(self):
        assert pc.adaptive_kl_beta(0.15, 0.05) == 0.2
        assert pc.adaptive_kl_beta(1.5e-4, 0.001) == pytest.approx(1e-4)
        with pytest.raises(pc.Phase9ContractError):
            pc.adaptive_kl_beta(0.0, 0.01)

    def test_entropy_schedule(self):
        assert pc.entropy_coefficient(1, 60) == 0.005
        assert pc.entropy_coefficient(60, 60) == 0.001
        assert pc.entropy_coefficient(1, 8) == 0.005
        assert pc.entropy_coefficient(8, 8) == 0.001
        middle = pc.entropy_coefficient(30, 60)
        assert 0.001 < middle < 0.005
        assert middle == pytest.approx(0.005 + (29 / 59) * (0.001 - 0.005))
        values = [pc.entropy_coefficient(i, 60) for i in range(1, 61)]
        assert values == sorted(values, reverse=True)
        with pytest.raises(pc.Phase9ContractError):
            pc.entropy_coefficient(0, 60)
        with pytest.raises(pc.Phase9ContractError):
            pc.entropy_coefficient(61, 60)

    def test_behavior_storage_tolerances(self):
        assert pc.BEHAVIOR_TEMPERATURE == 1.0
        assert pc.BEHAVIOR_PROBABILITY_ABS_TOLERANCE == 1e-4
        semantics = pc.behavior_policy_semantics()
        assert semantics["storage"]["trajectory_version"].startswith("trajectory_v1")
        assert "old_probabilities" in semantics["storage"]["stored_quantity"]
        assert semantics["verification"]["max_abs_mismatch"] == 1e-4
        assert "one-hot" in semantics["storage"]["opponent_rule_policy_representation"]


class TestPilotMatrix:
    def test_exactly_six_candidates(self):
        matrix = pc.pilot_matrix()
        assert len(matrix["candidates"]) == 6
        assert matrix["candidate_limit"] == 6
        assert matrix["no_seventh_run"] is True
        assert matrix["no_opportunistic_early_stop"] is True

    def test_the_frozen_grid(self):
        expected = {
            ("P9-A", 1e-4, 0.005),
            ("P9-B", 1e-4, 0.020),
            ("P9-C", 3e-4, 0.005),
            ("P9-D", 3e-4, 0.020),
            ("P9-E", 6e-4, 0.005),
            ("P9-F", 6e-4, 0.020),
        }
        observed = {
            (c["candidate_id"], c["learning_rate"], c["initial_kl_beta"])
            for c in pc.PILOT_CANDIDATES
        }
        assert observed == expected

    def test_namespaces_align_with_the_seed_module(self):
        assert tuple(c["namespace"] for c in pc.PILOT_CANDIDATES) == ps.PILOT_NAMESPACES

    def test_pilot_budget(self):
        budget = pc.pilot_matrix()["per_candidate_budget"]
        assert budget["rl_iterations"] == 8
        assert budget["games_per_iteration"] == 1024
        assert budget["optimizer_epochs_per_rollout"] == 2
        assert budget["mixture"] == pc.PILOT_BUCKET_COUNTS

    def test_hard_vetoes(self):
        vetoes = pc.PILOT_HARD_VETOES
        assert vetoes["illegal_neural_action_max"] == 0
        assert vetoes["non_finite_loss_max"] == 0
        assert vetoes["non_finite_gradient_max"] == 0
        assert vetoes["non_finite_parameter_max"] == 0
        assert vetoes["behavior_identity_mismatch_max"] == 0
        assert vetoes["target_reconstruction_mismatch_max"] == 0
        assert vetoes["observer_safety_failure_max"] == 0
        assert vetoes["checkpoint_resume_failure_max"] == 0
        assert vetoes["mean_iteration_or_epoch_kl_max"] == 0.08
        assert vetoes["iteration_ppo_clip_fraction_max"] == 0.75
        assert vetoes["validation_random_ewr_min"] == 0.90
        assert vetoes["validation_basic_ewr_min"] == 0.60

    def test_final_test_results_are_forbidden_evidence(self):
        assert "final-test results" in pc.pilot_matrix()["selection"]["forbidden_evidence"]


class TestValidationScore:
    def test_weights(self):
        assert pc.VALIDATION_SCORE_WEIGHTS == {
            "strategic_rule_based": 0.45,
            "tactical_rule_based": 0.35,
            "phase8_anchor": 0.20,
        }
        assert sum(pc.VALIDATION_SCORE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_arithmetic(self):
        assert pc.validation_score(1.0, 0.0, 0.0) == pytest.approx(0.45)
        assert pc.validation_score(0.0, 1.0, 0.0) == pytest.approx(0.35)
        assert pc.validation_score(0.0, 0.0, 1.0) == pytest.approx(0.20)
        assert pc.validation_score(0.5, 0.5, 0.5) == pytest.approx(0.5)
        assert pc.validation_score(0.52, 0.48, 0.5) == pytest.approx(
            0.45 * 0.52 + 0.35 * 0.48 + 0.20 * 0.5
        )

    def test_rejects_out_of_range(self):
        with pytest.raises(pc.Phase9ContractError):
            pc.validation_score(1.2, 0.5, 0.5)
        with pytest.raises(pc.Phase9ContractError):
            pc.validation_score(0.5, -0.1, 0.5)

    def test_tie_break_chain(self):
        assert pc.VALIDATION_TIE_BREAK == (
            "higher validation score",
            "higher Strategic EWR",
            "lower mean behavior KL",
            "higher training examples/s",
        )

    def test_regression_guards_are_not_score_components(self):
        assert pc.VALIDATION_REGRESSION_GUARDS == {
            "random_legal_ewr_min": 0.90,
            "basic_heuristic_ewr_min": 0.60,
        }
        assert "random" not in " ".join(pc.VALIDATION_SCORE_WEIGHTS)
        assert "basic" not in " ".join(pc.VALIDATION_SCORE_WEIGHTS)


class TestFinalGates:
    def test_gate_arithmetic(self):
        gates = pc.final_gates()
        gate_a = gates["gate_a_direct_improvement_over_anchor"]
        assert gate_a["paired_cases"] == 512
        assert gate_a["games"] == 1024
        assert gate_a["effective_win_rate_min"] == 0.58
        assert gate_a["paired_bootstrap_lower_bound_exclusive"] == 0.53
        for name in ("gate_b_strategic", "gate_c_tactical"):
            gate = gates[name]
            assert gate["final_ewr_min"] == 0.52
            assert gate["paired_improvement_over_anchor_min"] == 0.05
            assert gate["improvement_ci_lower_bound_exclusive"] == 0.0
        assert gates["stretch_report_only"] == {
            "strategic_ewr": 0.55,
            "tactical_ewr": 0.55,
        }
        gate_d = gates["gate_d_random_guard"]
        assert gate_d["overall_ewr_min"] == 0.94
        assert gate_d["red_ewr_min"] == 0.90
        assert gate_d["blue_ewr_min"] == 0.90
        assert gate_d["paired_bootstrap_lower_bound_exclusive"] == 0.92
        gate_e = gates["gate_e_basic_guard"]
        assert gate_e["ewr_min"] == 0.65
        assert gate_e["paired_bootstrap_lower_bound_exclusive"] == 0.60
        gate_f = gates["gate_f_safety"]
        assert all(value == 0 for value in gate_f.values())
        gate_g = gates["gate_g_policy_collapse"]
        assert gate_g["max_legal_probability_threshold"] == 0.999
        assert gate_g["fraction_above_threshold_max_exclusive"] == 0.25
        gate_h = gates["gate_h_belief_retention"]
        assert gate_h["belief_ce_ratio_vs_remaining_count_max"] == 0.98
        assert gate_h["belief_top1_must_beat_remaining_count_top1"] is True

    def test_statistics_are_frozen(self):
        statistics = pc.final_gates()["statistics"]
        assert statistics["method"] == "paired_unit_percentile_bootstrap"
        assert statistics["replicates"] == 10000
        assert statistics["confidence"] == 0.95
        assert statistics["bootstrap_seed"] == 2026081608

    def test_report_only_cannot_rescue(self):
        assert "may not rescue" in pc.final_gates()["report_only_rule"]


class TestCanonicalRunBudget:
    def test_budget(self):
        run = pc.canonical_run_contract()
        assert run["rl_iterations"] == 60
        assert run["games_per_iteration"] == 2048
        assert run["max_scheduled_games"] == 122880
        assert run["optimizer_epochs_per_rollout"] == 2
        assert run["validation_cadence_iterations"] == 5
        assert run["archive_cadence_iterations"] == 5
        assert run["wall_clock_ceiling_hours"] == 12


class TestSealing:
    def test_structural_audit_is_always_allowed(self):
        for agent in range(1, 9):
            access = pc.check_test_bank_access("structural_audit", phase9_agent=agent)
            assert access.resource == "phase9_test_bank"

    def test_neural_purposes_are_refused_before_agent_8(self):
        for agent in range(1, 8):
            for purpose in (
                "neural_model_inference",
                "model_metric",
                "checkpoint_selection",
                "hyperparameter_selection",
                "final_evaluation",
            ):
                with pytest.raises(pc.Phase9SealingError):
                    pc.check_test_bank_access(purpose, phase9_agent=agent)

    def test_agent_8_final_evaluation_is_allowed(self):
        access = pc.check_test_bank_access("final_evaluation", phase9_agent=8)
        assert access.phase9_agent == 8

    def test_agent_8_still_cannot_use_prohibited_purposes(self):
        for purpose in pc.TEST_BANK_PROHIBITED_BEFORE_8:
            with pytest.raises(pc.Phase9SealingError):
                pc.check_test_bank_access(purpose, phase9_agent=8)

    def test_unknown_purposes_and_agents_are_refused(self):
        with pytest.raises(pc.Phase9SealingError):
            pc.check_test_bank_access("casual_look", phase9_agent=1)
        with pytest.raises(pc.Phase9SealingError):
            pc.check_test_bank_access("structural_audit", phase9_agent=9)

    def test_validation_bank_selection_is_allowed_to_every_agent(self):
        for agent in range(1, 9):
            for purpose in (
                "structural_audit",
                "anchor_baseline",
                "validation_scoring",
                "pilot_selection",
                "checkpoint_selection",
            ):
                access = pc.check_validation_bank_access(purpose, phase9_agent=agent)
                assert access.resource == "phase9_validation_bank"

    def test_validation_bank_never_updates_weights(self):
        for agent in range(1, 9):
            with pytest.raises(pc.Phase9SealingError):
                pc.check_validation_bank_access("weight_update", phase9_agent=agent)

    def test_no_test_metric_before_agent_8_is_declared(self):
        assert pc.sealing_rules()["no_test_metric_before_agent_8"] is True


class TestCheckpointContract:
    def test_required_fields_cover_the_common_contract(self):
        fields = set(pc.CHECKPOINT_REQUIRED_FIELDS)
        for required in (
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "global_optimizer_step",
            "rl_iteration",
            "minibatch_cursor",
            "examples_consumed",
            "behavior_snapshot_identity",
            "behavior_checkpoint_sha256",
            "rollout_iteration_identity",
            "sealed_rollout_digest",
            "kl_beta",
            "kl_controller_state",
            "entropy_schedule_position",
            "population_version",
            "active_historical_identities",
            "historical_checkpoint_digests",
            "opponent_schedule_version",
            "setup_sampler_version",
            "best_validation_score",
            "best_checkpoint_identity",
            "validation_history",
            "phase9_seeds",
            "corpus_identities",
            "rules_model_observation_versions",
            "wall_clock_counters",
            "software_runtime_versions",
        ):
            assert required in fields, required

    def test_rollout_lifecycle(self):
        assert pc.ROLLOUT_STATES == (
            "COLLECTING", "SEALED", "TRAINING", "EVALUATED", "COMMITTED",
        )
        store = pc.rollout_store_schema()
        assert store["states"] == list(pc.ROLLOUT_STATES)
        assert "one iteration must never mix two behavior snapshot identities" in (
            store["crash_rules"]
        )


class TestEvalBankContract:
    def test_bank_shapes(self):
        assert pc.VALIDATION_BANK_CASES == 128
        assert pc.TEST_BANK_CASES == 512
        assert pc.VALIDATION_CASES_PER_FAMILY == 8
        assert pc.TEST_CASES_PER_FAMILY == 32
        assert pc.SETUP_FAMILY_COUNT == 16
        contract = pc.eval_bank_contract()
        assert contract["banks"]["validation"]["bootstrap_seed"] == 2026081607
        assert contract["banks"]["test"]["bootstrap_seed"] == 2026081608
        assert contract["pairing_mode"] == "color_swap_same_board"

    def test_core_opponents(self):
        assert pc.CORE_OPPONENTS == (
            "phase8_anchor",
            "random_legal",
            "basic_heuristic",
            "tactical_rule_based",
            "strategic_rule_based",
        )

    def test_stress_schedule_is_smaller_and_report_only(self):
        contract = pc.eval_bank_contract()["stress_schedule"]
        assert contract["validation_pairs_per_policy"] == 32
        assert contract["test_pairs_per_policy"] == 64
        assert contract["validation_pairs_per_policy"] < pc.VALIDATION_BANK_CASES
        assert contract["test_pairs_per_policy"] < pc.TEST_BANK_CASES
        assert "report-only" in contract["validation_rule"]
        assert "report-only" in contract["test_rule"]


def test_entropy_coefficient_never_leaves_the_frozen_interval():
    for total in (8, 60):
        for iteration in range(1, total + 1):
            value = pc.entropy_coefficient(iteration, total)
            assert 0.001 <= value <= 0.005
            assert math.isfinite(value)
