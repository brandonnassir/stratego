"""Phase 18 Gate G2: the machine-readable S01-S30 coverage table.

Every row of `reports/phase18/ataraxos_setup_method_map_v2.json` that the
setup parity build owns is mapped here to the implementation symbols that
realise it and to the test functions that prove it. `verify_coverage`
resolves every symbol by import and every test by parsing the test file, so
a row cannot cite a symbol or a test that does not exist; the driver then
attaches the recorded pytest outcome of every cited test, so a row is
`complete` only when its tests ran and passed in the recorded run -- never on
documentation alone.

Rows S31-S35 are owned by later agents (tandem pilot, production rehearsal)
and are listed as out of scope for G2 with no completion claimed.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

PACKAGE = "stratego.training.phase18"
TEST_DIRECTORY = "tests/training/phase18"

_M = "test_setup_model.py"
_S = "test_setup_sampling.py"
_B = "test_setup_buffer.py"
_L = "test_setup_learning.py"
_O = "test_reference_oracle.py"
_A = "test_synthetic_assay.py"

#: Row -> implementation symbols (`module:Qualified.name`) and tests
#: (`file::function`). Parametrised tests are cited by their base name.
COVERAGE: dict = {
    "S01": {
        "implementation": ["setup_model:Phase18SetupModel.forward", "setup_model:CausalSelfAttention.forward"],
        "tests": [
            f"{_M}::test_prefix_k_is_a_function_of_the_first_k_placements_only",
            f"{_M}::test_prefix_k_has_seen_exactly_k_placements",
            f"{_M}::test_every_head_reports_at_all_forty_prefixes",
            f"{_O}::test_masks_and_information_and_expected_values_and_aggregation_and_advantage_match",
        ],
    },
    "S02": {
        "implementation": ["setup_sampling:remaining_counts", "setup_sampling:batched_remaining", "setup_sampling:legal_masks", "setup_sampling:masked_log_probabilities"],
        "tests": [
            f"{_S}::test_exhausted_types_receive_probability_exactly_zero",
            f"{_S}::test_after_all_eight_scouts_are_placed_the_scout_probability_is_exactly_zero",
            f"{_S}::test_every_generated_setup_reproduces_the_classic_piece_counts",
            f"{_S}::test_recorded_masks_are_inventory_and_handedness_and_nothing_is_passed_in",
            f"{_S}::test_batched_remaining_reads_only_the_prefix_columns",
            f"{_S}::test_an_over_used_type_raises_rather_than_being_repaired",
        ],
    },
    "S03": {
        "implementation": ["setup_model:Phase18SetupModel", "setup_contract:SETUP_VOCABULARY"],
        "tests": [f"{_M}::test_twelve_way_softmax_equals_the_published_fourteen_way_softmax_restricted_to_live_classes"],
    },
    "S04": {
        "implementation": ["setup_sampling:handedness_mask", "setup_sampling:legal_masks", "setup_contract:FLAG_PERMITTED_FILES"],
        "tests": [
            f"{_S}::test_the_handedness_mask_forbids_the_flag_on_the_left_five_files_only",
            f"{_S}::test_forced_generation_puts_the_flag_in_the_permitted_half_100_percent",
            f"{_S}::test_reduction_without_forced_handedness_reproduces_the_unconstrained_distribution",
        ],
    },
    "S05": {
        "implementation": ["setup_sampling:reflect_tokens", "setup_sampling:generate_pool", "setup_contract:reflection_seed"],
        "tests": [
            f"{_S}::test_reflection_is_an_involution_and_matches_the_accepted_helper",
            f"{_S}::test_reflected_fraction_is_one_half_within_binomial_tolerance",
            f"{_S}::test_the_reflection_stream_is_independent_of_the_token_stream",
            f"{_S}::test_a_reflected_sample_plays_the_mirror_and_keeps_its_network_record",
        ],
    },
    "S06": {
        "implementation": ["setup_buffer:SetupBuffer._batch", "setup_sampling:SampledSetup"],
        "tests": [
            f"{_S}::test_flipping_a_played_board_back_recovers_the_network_tokens_and_the_recorded_nll",
            f"{_S}::test_gathering_against_the_played_orientation_would_be_wrong_for_reflected_samples",
            f"{_B}::test_minibatches_flip_played_boards_back_and_refuse_a_corrupted_record",
        ],
    },
    "S07": {
        "implementation": ["setup_sampling:to_engine_setup"],
        "tests": [
            f"{_S}::test_red_is_the_identity_and_blue_reverses_the_ranks",
            f"{_S}::test_canonical_blue_handed_straight_to_the_engine_is_rejected",
            f"{_S}::test_blue_flags_land_on_blue_back_rows_not_front_rows",
            f"{_S}::test_an_illegal_inventory_or_unknown_player_never_reaches_the_engine",
            f"{_S}::test_every_pooled_board_is_accepted_by_engine_game_creation_without_a_move_played",
        ],
    },
    "S08": {
        "implementation": ["setup_buffer:expected_value", "setup_buffer:outcome_one_hot", "setup_contract:CATEGORICAL_AGGREGATION"],
        "tests": [
            f"{_B}::test_category_order_is_loss_draw_win_and_the_aggregation_vector_is_minus_one_zero_one",
            f"{_B}::test_value_logits_are_stored_and_softmaxed_when_they_enter_the_advantage",
        ],
    },
    "S09": {
        "implementation": ["setup_buffer:SetupBuffer.add_outcome", "setup_buffer:SetupBuffer.process"],
        "tests": [
            f"{_B}::test_a_known_multiset_of_outcomes_aggregates_to_the_closed_form",
            f"{_B}::test_a_setup_with_zero_outcomes_is_excluded_not_trained_as_a_draw",
            f"{_B}::test_a_period_with_no_outcomes_at_all_refuses_rather_than_inventing_draws",
            f"{_O}::test_masks_and_information_and_expected_values_and_aggregation_and_advantage_match",
        ],
    },
    "S10": {
        "implementation": ["setup_buffer:mark_most_recent_appearance", "setup_buffer:SetupBuffer.add_pool", "setup_buffer:SetupBuffer.index_of"],
        "tests": [
            f"{_B}::test_identical_played_boards_collapse_to_one_row_bound_to_the_newer_snapshot",
            f"{_B}::test_mark_most_recent_appearance_matches_the_published_helper",
            f"{_B}::test_an_outcome_for_an_unknown_setup_is_fatal_never_dropped",
        ],
    },
    "S11": {
        "implementation": ["setup_sampling:suffix_information"],
        "tests": [
            f"{_S}::test_suffix_information_is_the_reverse_cumulative_surprisal",
            f"{_S}::test_information_recursion_holds_on_recorded_samples",
        ],
    },
    "S12": {
        "implementation": ["setup_buffer:SetupBuffer.process", "setup_learning:setup_batch_loss"],
        "tests": [f"{_L}::test_the_entropy_target_is_i_over_ten_and_the_head_converges_to_it_not_to_i"],
    },
    "S13": {
        "implementation": ["setup_buffer:SetupBuffer.process", "setup_contract:ENTROPY_NORMALIZER"],
        "tests": [
            f"{_B}::test_the_residual_is_exactly_zero_when_h_equals_i_over_ten",
            f"{_B}::test_the_centered_form_averages_to_zero_where_phase17s_form_leaves_nine_tenths_of_i",
            f"{_B}::test_the_processed_advantage_uses_ten_times_h",
            f"{_B}::test_advantage_terms_are_reported_separately",
            f"{_O}::test_i_minus_ten_h_in_the_oracle_is_the_published_denormalisation",
        ],
    },
    "S14": {
        "implementation": ["setup_contract:setup_alpha"],
        "tests": [f"{_B}::test_alpha_is_the_published_power_schedule_and_no_clamp_binds_within_the_horizon"],
    },
    "S15": {
        "implementation": ["setup_buffer:SetupBuffer.process", "reference_oracle:oracle_published_recursion", "setup_contract:SetupTrainingConfig"],
        "tests": [
            f"{_B}::test_the_published_recursion_at_lambda_one_equals_the_flat_form",
            f"{_B}::test_a_lambda_other_than_one_is_refused_rather_than_silently_computed",
        ],
    },
    "S16": {
        "implementation": ["setup_learning:setup_batch_loss", "setup_contract:SETUP_PPO_CLIP_EPSILON"],
        "tests": [
            f"{_L}::test_a_ratio_of_one_gives_minus_the_mean_advantage",
            f"{_L}::test_a_positive_advantage_with_ratio_above_1_2_is_clipped_and_a_negative_below_0_8_is_clipped",
            f"{_O}::test_the_oracle_ppo_clips_in_both_directions_and_the_kl_direction_is_current_given_behavior",
        ],
    },
    "S17": {
        "implementation": ["setup_learning:setup_batch_loss", "setup_contract:SETUP_KL_DIRECTION"],
        "tests": [
            f"{_L}::test_the_kl_is_current_given_behavior_and_illegal_types_contribute_exactly_zero",
            f"{_O}::test_the_oracle_ppo_clips_in_both_directions_and_the_kl_direction_is_current_given_behavior",
        ],
    },
    "S18": {
        "implementation": ["setup_contract:SETUP_VALUE_LOSS_WEIGHT", "setup_contract:SETUP_ENTROPY_PREDICTION_LOSS_WEIGHT", "setup_contract:SETUP_BEHAVIOR_KL_COEFFICIENT", "setup_learning:setup_batch_loss"],
        "tests": [
            f"{_L}::test_the_four_coefficients_are_frozen_and_the_total_is_their_weighted_sum",
            f"{_O}::test_every_loss_term_and_the_total_match_the_oracle_in_double_precision",
        ],
    },
    "S19": {
        "implementation": ["setup_buffer:SetupBuffer.minibatches"],
        "tests": [f"{_B}::test_the_batch_carries_forty_prefix_rows_per_ready_setup_with_no_mask"],
    },
    "S20": {
        "implementation": ["setup_sampling:generate_pool", "setup_contract:SETUP_POOL_SIZE"],
        "tests": [f"{_S}::test_a_pool_of_1024_has_1024_distinct_entries_split_512_per_lane"],
    },
    "S21": {
        "implementation": ["setup_buffer:SetupBuffer.filter"],
        "tests": [
            f"{_B}::test_an_outcome_finishing_under_the_next_pool_attributes_to_the_old_row",
            f"{_B}::test_an_undersized_window_raises_rather_than_dropping_the_outcome",
        ],
    },
    "S22": {
        "implementation": ["setup_buffer:SetupBuffer.process", "setup_sampling:SampledSetup"],
        "tests": [f"{_B}::test_the_advantage_uses_recorded_behavior_quantities_not_a_reforward"],
    },
    "S23": {
        "implementation": ["setup_buffer:SetupBuffer.add_pool"],
        "tests": [f"{_B}::test_counts_and_ready_flags_reset_when_a_new_pool_arrives"],
    },
    "S24": {
        "implementation": ["setup_sampling:has_opening_move", "synthetic_assay:run_seed"],
        "tests": [
            f"{_S}::test_opening_move_predicate_matches_the_engine_at_ply_zero",
            f"{_S}::test_a_terminal_setup_is_flagged_by_the_sample_and_counted_by_the_pool",
            f"{_A}::test_the_reduced_run_writes_every_artifact_with_zero_integrity_events",
        ],
    },
    "S25": {
        "implementation": ["setup_learning:SetupTrainer.__init__", "setup_contract:SETUP_WEIGHT_DECAY"],
        "tests": [
            f"{_L}::test_adam_and_adamw_at_zero_weight_decay_take_identical_steps",
            f"{_L}::test_the_trainer_uses_adamw_with_zero_decay_and_the_config_refuses_a_nonzero_decay",
            f"{_O}::test_torch_adamw_at_zero_decay_matches_the_oracle_adam_update",
        ],
    },
    "S26": {
        "implementation": ["setup_learning:SetupTrainer.update", "setup_contract:SETUP_BATCH_SIZE", "setup_contract:SETUP_EPOCHS_PER_UPDATE"],
        "tests": [
            f"{_L}::test_one_optimizer_step_per_minibatch_per_epoch_and_one_ema_update",
            f"{_L}::test_the_production_batch_is_1024_setups_which_is_one_step_per_epoch_at_1024_ready",
            f"{_O}::test_step_counts_and_ema_cadence_follow_the_published_loop",
        ],
    },
    "S27": {
        "implementation": ["setup_learning:SetupTrainer.update", "setup_contract:SETUP_GRADIENT_CLIP_NORM"],
        "tests": [
            f"{_L}::test_the_post_clip_norm_never_exceeds_0_5_and_only_setup_parameters_are_stepped",
            f"{_O}::test_clip_grad_norm_scale_matches_the_oracle",
        ],
    },
    "S28": {
        "implementation": ["setup_learning:SetupEMA", "setup_learning:SetupTrainer.generation_actor", "setup_learning:SetupTrainer.evaluation_model"],
        "tests": [
            f"{_L}::test_the_ema_follows_the_closed_form_and_is_updated_once_per_update",
            f"{_L}::test_the_raw_model_generates_and_the_ema_only_evaluates",
        ],
    },
    "S29": {
        "implementation": ["setup_learning:SetupTrainer.save_checkpoint", "setup_learning:SetupTrainer.load_checkpoint", "setup_learning:SetupEMA.load_state_dict"],
        "tests": [
            f"{_L}::test_save_reload_and_one_more_update",
            f"{_L}::test_a_tampered_checkpoint_file_is_refused",
        ],
    },
    "S30": {
        "implementation": ["setup_model:assert_architecture", "setup_contract:SETUP_PARAMETER_TARGET"],
        "tests": [
            f"{_M}::test_parameter_count_is_exactly_802320",
            f"{_M}::test_architecture_is_4_128_4_512_pre_layernorm",
            f"{_M}::test_positional_embeddings_initialise_at_std_0_1",
            f"{_M}::test_a_different_width_is_refused_rather_than_accepted_quietly",
            f"{_M}::test_state_dict_shapes_match_the_accepted_phase17_architecture",
        ],
    },
}

OUT_OF_SCOPE = {
    "S31": "tandem pilot agent: move-policy training regime (no self-play in Phase 18)",
    "S32": "tandem pilot agent: on-policy requirement for setup PPO data (live stream only)",
    "S33": "tandem pilot agent: canonical/live mixture and cadence (provisional)",
    "S34": "tandem pilot agent: belief and search separation",
    "S35": "production rehearsal agent: signal handling and clean shutdown",
}

G2_ROWS = tuple(f"S{index:02d}" for index in range(1, 31))


def _resolve(symbol: str) -> bool:
    module_name, _, qualified = symbol.partition(":")
    module = importlib.import_module(f"{PACKAGE}.{module_name}")
    target = module
    for part in qualified.split("."):
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


def _test_functions(path: Path) -> set:
    tree = ast.parse(path.read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}


def verify_coverage(repository_root) -> dict:
    """Resolve every cited symbol and test; report the problems, if any."""
    root = Path(repository_root)
    method_map = json.loads((root / "reports/phase18/ataraxos_setup_method_map_v2.json").read_text())
    rows_by_id = {row["id"]: row for row in method_map["rows"]}
    problems: list = []
    functions_cache: dict = {}
    rows: dict = {}
    for row_id in G2_ROWS:
        entry = COVERAGE.get(row_id)
        if entry is None:
            problems.append(f"{row_id}: no coverage entry")
            continue
        if row_id not in rows_by_id:
            problems.append(f"{row_id}: not in the method map")
            continue
        if not entry["implementation"] or not entry["tests"]:
            problems.append(f"{row_id}: a row needs at least one implementation symbol and one test")
        for symbol in entry["implementation"]:
            try:
                if not _resolve(symbol):
                    problems.append(f"{row_id}: symbol {symbol} does not resolve")
            except Exception as error:  # import failure
                problems.append(f"{row_id}: symbol {symbol} failed to import: {error}")
        for test in entry["tests"]:
            file_name, _, function = test.partition("::")
            path = root / TEST_DIRECTORY / file_name
            if path not in functions_cache:
                functions_cache[path] = _test_functions(path) if path.exists() else set()
            if function not in functions_cache[path]:
                problems.append(f"{row_id}: test {test} does not exist")
        rows[row_id] = {
            "element": rows_by_id[row_id]["element"],
            "map_status": rows_by_id[row_id]["status"],
            "required_test": rows_by_id[row_id]["test"],
            "implementation": list(entry["implementation"]),
            "tests": list(entry["tests"]),
        }
    for row_id in sorted(set(COVERAGE) - set(G2_ROWS)):
        problems.append(f"{row_id}: coverage entry outside the G2 rows")
    return {
        "package": PACKAGE,
        "test_directory": TEST_DIRECTORY,
        "rows": rows,
        "out_of_scope": {row_id: {"element": rows_by_id[row_id]["element"], "owner": rows_by_id[row_id]["owner"], "note": note, "complete": False} for row_id, note in OUT_OF_SCOPE.items()},
        "problems": problems,
        "verified": not problems,
    }


def attach_test_outcomes(coverage: dict, outcomes: dict) -> dict:
    """Mark every row from the recorded pytest outcomes.

    `outcomes` maps `file::function` to a status among `passed`, `failed`,
    `skipped`, `error`, `missing`; a parametrised test contributes its worst
    case. A row is `complete` only when every cited test passed.
    """
    for row_id, row in coverage["rows"].items():
        statuses = {test: outcomes.get(test, "missing") for test in row["tests"]}
        row["test_outcomes"] = statuses
        row["complete"] = all(status == "passed" for status in statuses.values())
        row["status"] = "complete" if row["complete"] else "NOT complete"
    coverage["rows_complete"] = sum(1 for row in coverage["rows"].values() if row["complete"])
    coverage["rows_total"] = len(coverage["rows"])
    coverage["all_g2_rows_complete"] = coverage["rows_complete"] == coverage["rows_total"] == len(G2_ROWS) and coverage["verified"]
    return coverage


__all__ = ["COVERAGE", "G2_ROWS", "OUT_OF_SCOPE", "attach_test_outcomes", "verify_coverage"]
