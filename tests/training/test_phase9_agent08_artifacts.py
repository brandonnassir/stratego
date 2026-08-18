"""Phase 9 Agent 8: the published final-acceptance artifacts stay honest.

These tests read the reports rather than recompute them. Their job is to stop
a published acceptance from drifting away from the frozen evaluation it
claims to record — a hard-gate boolean that contradicts its own observed and
threshold numbers, a recommendation that ignores a failed gate, a strength
table whose game counts are not the frozen 512-pair schedule, a league matrix
that skips iterations, or an identity block whose digests are not the
accepted ones.

The artifacts exist only after `scripts/run_phase9_agent08.py` has run, so
every test skips cleanly when they are absent rather than failing a fresh
checkout.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from stratego.training.phase9_amendment import amendment_digest
from stratego.training.phase9_amendment_v2 import (
    amendment_digest as amendment_v2_digest,
    verify_chain_untouched,
)
from stratego.training.phase9_contract import (
    CANONICAL_ITERATIONS,
    STRESS_POLICY_ROSTER,
    TEST_BANK_CASES,
    TEST_STRESS_PAIRS,
    contract_digest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"
ACCEPTANCE_PATH = DATA_DIRECTORY / "agent_08_final_acceptance.json"
STRENGTH_PATH = DATA_DIRECTORY / "agent_08_strength_results.csv"
LEAGUE_PATH = DATA_DIRECTORY / "agent_08_league_matrix.csv"
HARNESS_PATH = REPOSITORY_ROOT / "scripts" / "run_phase9_agent08.py"

pytestmark = pytest.mark.skipif(
    not ACCEPTANCE_PATH.exists(),
    reason="Agent 8 artifacts not generated yet (run scripts/run_phase9_agent08.py)",
)

RULE_MATCHUPS = (
    "candidate_vs_random_legal",
    "candidate_vs_basic_heuristic",
    "candidate_vs_tactical_rule_based",
    "candidate_vs_strategic_rule_based",
    "candidate_vs_phase8_anchor",
    "anchor_vs_strategic_rule_based",
    "anchor_vs_tactical_rule_based",
)


@pytest.fixture(scope="module")
def acceptance() -> dict:
    with open(ACCEPTANCE_PATH, "r", encoding="utf-8") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def harness():
    specification = importlib.util.spec_from_file_location(
        "run_phase9_agent08_artifact_tests", HARNESS_PATH
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def strength_rows() -> list:
    with open(STRENGTH_PATH, "r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_recommendation_is_a_frozen_word(acceptance):
    assert acceptance["recommendation"] in ("PASS", "FAIL", "BLOCKED")
    assert acceptance["status"] == acceptance["recommendation"]


def test_hard_gate_table_is_self_consistent(acceptance, harness):
    # The same validator the writer ran; a published table whose booleans
    # disagree with its own numbers can never survive this.
    assert harness.validate_acceptance_artifact(acceptance) == []


def test_hard_gate_table_names_and_shape(acceptance, harness):
    table = acceptance["hard_gates"]
    assert sorted(table) == sorted(name for name, _key in harness.HARD_GATE_ROWS)
    for row in table.values():
        assert set(("observed", "threshold", "passed")) <= set(row)


def test_frozen_thresholds_are_the_contract_numbers(acceptance):
    table = acceptance["hard_gates"]
    assert table["gate_a_vs_phase8_anchor"]["threshold"] == {
        "effective_win_rate_min": 0.58,
        "paired_bootstrap_lower_exclusive": 0.53,
    }
    for name in ("gate_b_strategic", "gate_c_tactical"):
        assert table[name]["threshold"] == {
            "effective_win_rate_min": 0.52,
            "paired_improvement_min": 0.05,
            "improvement_ci_lower_exclusive": 0.0,
        }
    assert table["gate_d_random_guard"]["threshold"] == {
        "overall_ewr_min": 0.94,
        "red_ewr_min": 0.90,
        "blue_ewr_min": 0.90,
        "paired_bootstrap_lower_exclusive": 0.92,
    }
    assert table["gate_e_basic_guard"]["threshold"] == {
        "ewr_min": 0.65,
        "paired_bootstrap_lower_exclusive": 0.60,
    }
    assert table["gate_g_policy_collapse"]["threshold"] == {
        "fraction_above_0_999_max_exclusive": 0.25
    }
    assert table["gate_h_belief_retention"]["threshold"] == {
        "belief_ce_ratio_max": 0.98,
        "belief_top1_must_beat_remaining_count_top1": True,
    }


def test_frozen_identity_block_matches_the_live_modules(acceptance):
    frozen = acceptance["frozen_inputs"]
    assert frozen["contract_digest"] == contract_digest()
    assert frozen["amendment_v1_digest"] == amendment_digest()
    assert frozen["amendment_v2_digest"] == amendment_v2_digest()
    assert frozen["ceiling_chain_hours"] == [12, 15, 24]
    assert verify_chain_untouched() == []
    assert frozen["selected_iteration"] == 40
    assert frozen["source_snapshot"] == "behavior_B041.pt"
    assert frozen["phase9_checkpoint_sha256"].startswith("dfd698e5")
    assert frozen["phase9_model_state_digest"].startswith("f1df694d")
    assert frozen["phase8_checkpoint_sha256"].startswith("f7e9c40d")


def test_completion_gates_include_the_required_names(acceptance):
    required = {
        "agents1_7_pass",
        "corpus_resolver_verified",
        "corpus_digests_match",
        "phase8_checkpoint_verified",
        "phase9_checkpoint_verified",
        "phase9_config_verified",
        "final_bank_verified",
        "pre_agent8_final_test_access_zero",
        "phase9_vs_phase8_gate",
        "strategic_gate",
        "tactical_gate",
        "random_gate",
        "basic_gate",
        "belief_retention_gate",
        "collapse_gate",
        "illegal_actions_zero",
        "model_failures_zero",
        "nonfinite_outputs_zero",
        "observer_safety_zero",
        "paired_bootstrap_exact",
        "report_only_diagnostics_written",
        "full_suite_green",
    }
    assert required <= set(acceptance["completion_gates"])
    assert acceptance["gates_total"] == len(acceptance["completion_gates"])
    assert acceptance["gates_true"] == sum(
        1 for value in acceptance["completion_gates"].values() if value
    )


def test_final_summary_records_the_sealed_access(acceptance):
    access = acceptance["final_summary"]["authorized_access"]
    assert access == {
        "resource": "phase9_test_bank",
        "purpose": "final_evaluation",
        "phase9_agent": 8,
    }


def test_protocol_is_the_frozen_one(acceptance):
    protocol = acceptance["final_summary"]["protocol"]
    assert protocol["decision_mode"] == "greedy"
    assert protocol["batch_policy"] == "single_request"
    assert protocol["dtype"] == "float32"
    assert protocol["pairing_mode"] == "color_swap_same_board"
    assert protocol["bootstrap_base_seed"] == 2026081608
    assert protocol["bank_version"] == "phase9_test_bank_v1"
    assert protocol["frozen_checkpoint_sha256"].startswith("dfd698e5")


def test_matchup_game_counts_are_the_frozen_schedule(acceptance):
    matchups = acceptance["final_summary"]["matchups"]
    for label in RULE_MATCHUPS:
        assert matchups[label]["games"] == TEST_BANK_CASES * 2, label
    for policy_id in STRESS_POLICY_ROSTER:
        assert matchups[f"candidate_vs_{policy_id}"]["games"] == TEST_STRESS_PAIRS * 2
    assert len(matchups) == len(RULE_MATCHUPS) + len(STRESS_POLICY_ROSTER)


def test_gate_population_covers_every_candidate_final_game(acceptance):
    replays = acceptance["final_summary"]["replays"]
    expected_labels = {
        label
        for label in acceptance["final_summary"]["matchups"]
        if label.startswith("candidate_vs_")
    }
    assert set(replays) == expected_labels
    total_games = sum(replay["games"] for replay in replays.values())
    candidate_games = sum(
        acceptance["final_summary"]["matchups"][label]["games"]
        for label in expected_labels
    )
    errored = sum(
        acceptance["final_summary"]["matchups"][label]["policy_errors"]
        for label in expected_labels
    )
    assert total_games == candidate_games - errored
    for replay in replays.values():
        assert replay["action_mismatches"] == 0
        assert replay["non_finite_policy_rows"] == 0
        assert replay["observer_failures"] == 0


def test_safety_zeros_agree_with_gate_f(acceptance):
    gate_f = acceptance["hard_gates"]["gate_f_safety"]
    safety = acceptance["final_summary"]["safety"]
    assert gate_f["observed"]["illegal_actions"] == safety["illegal_policy_actions"]
    if acceptance["recommendation"] == "PASS":
        assert gate_f["observed"] == {
            "illegal_actions": 0,
            "model_failures": 0,
            "non_finite_outputs": 0,
            "observer_safety_failures": 0,
        }


def test_observer_reconciliation_is_exact_and_documented(acceptance):
    observer = acceptance["discipline_summary"]["observer_reconciliation"]
    assert observer["reconciliation"]["reconstruction_is_exact"] is True
    assert observer["iteration_30"]["w00_committed_games"] == 27
    assert observer["iteration_30"]["w01_committed_games"] == 2021
    assert observer["probe_rule"]["iterations_checked"] == CANONICAL_ITERATIONS
    assert observer["probe_rule"]["iterations_exact"] == CANONICAL_ITERATIONS
    assert observer["probe_rule"]["mismatches"] == []
    assert observer["probe_replay"]["games_replayed"] == 27
    assert observer["probe_replay"]["failures"] == 0
    assert (
        observer["reconciliation"]["corrected_full_run_total"]
        == observer["reconciliation"]["recorded_session_total"]
        + observer["reconciliation"]["lost_session_probes_reconstructed"]
    )


def test_belief_retention_reports_the_phase8_benchmark(acceptance):
    belief = acceptance["final_summary"]["belief_retention"]
    assert belief["split"] == "test"
    assert belief["belief_pieces"] > 0
    assert belief["model_state_digest"].startswith("f1df694d")
    assert "baseline_rule" in belief
    gate_h = acceptance["hard_gates"]["gate_h_belief_retention"]
    assert gate_h["observed"]["belief_ce_ratio"] == belief["belief_ce_ratio"]
    assert gate_h["observed"]["belief_top1"] == belief["belief_top1"]


def test_paired_bootstrap_reproduction_is_exact(acceptance):
    check = acceptance["final_summary"]["paired_bootstrap_exact"]
    assert check["exact"] is True
    assert check["official_lower"] == check["independent_lower"]
    assert check["official_upper"] == check["independent_upper"]


def test_strength_csv_covers_every_matchup(acceptance, strength_rows):
    labels = {row["matchup"] for row in strength_rows}
    assert labels == set(acceptance["final_summary"]["matchups"])
    for row in strength_rows:
        matchup = acceptance["final_summary"]["matchups"][row["matchup"]]
        assert int(row["games"]) == matchup["games"]
        assert float(row["effective_win_rate"]) == matchup["effective_win_rate"]
        assert row["results_digest"] == matchup["results_digest"]


def test_league_matrix_covers_all_sixty_iterations():
    with open(LEAGUE_PATH, "r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    iterations = sorted({int(row["iteration"]) for row in rows})
    assert iterations == list(range(1, CANONICAL_ITERATIONS + 1))
    kinds = {row["opponent_kind"] for row in rows}
    assert {"current_policy", "rule_policy", "stress_policy"} <= kinds
    # The anchor plays throughout; archive members appear as the league grows.
    historical = [row for row in rows if row["opponent_kind"] == "historical_snapshot"]
    assert historical
    for row in rows:
        games = int(row["games"])
        assert games > 0
        assert int(row["red_wins"]) + int(row["blue_wins"]) + int(row["draws"]) == games


def test_tests_before_matches_the_pinned_pre_edit_suite(acceptance, harness):
    assert acceptance["tests_before"] == harness.TESTS_BEFORE


def test_recommendation_line_matches_gates(acceptance):
    if acceptance["recommendation"] == "PASS":
        assert acceptance["hard_gates_all_pass"] is True
        assert all(acceptance["completion_gates"].values())
