"""Phase 10 Agent 7's three artifacts, checked against the frozen contract.

The final-test evaluation is sealed and ran once; what these tests protect
is the record of it. Every claim below is recomputed from the artifacts'
own primitives — the classification from its own gate rows, the gate
booleans from their own observed values and the frozen thresholds, the CSV
rows against the acceptance JSON — so an edited number fails here rather
than being inherited by the Phase 10 closure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from stratego.training.phase10_acceptance import classify
from stratego.training.phase10_contract import (
    DIVERSITY_THRESHOLDS,
    GATE_A,
    GATE_B,
    HARD_GATE_IDS,
)
from stratego.training.phase10_seed import TEST_BOOTSTRAP_SEED

from .phase10_frozen_digests import BANK_DIGESTS, CONTRACT_BUNDLE_DIGEST, CONTRACT_DIGESTS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"
ACCEPTANCE_PATH = DATA_DIRECTORY / "agent_07_final_acceptance.json"
STRENGTH_PATH = DATA_DIRECTORY / "agent_07_strength_results.csv"
DIVERSITY_PATH = DATA_DIRECTORY / "agent_07_diversity_results.csv"
REPORT_PATH = REPOSITORY_ROOT / "reports" / "phase_10_implementation_report.md"

ACCEPTED_PHASE9_CHECKPOINT_SHA256 = (
    "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
)
ACCEPTED_PHASE9_MODEL_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)
ACCEPTED_SYSTEM_DIGEST = (
    "615cc3c3a4fab6e4400e20a5a93b13a08c43ab6c3ca63828c6a64742e98175d2"
)
ACCEPTED_CONFIG_SHA256 = (
    "6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668"
)

#: The 28 completion gates of the Agent 7 instruction, plus the suite gate.
EXPECTED_GATES = (
    "agents1_6_pass",
    "administrative_freeze_verified",
    "phase9_identity_verified",
    "phase7_identity_verified",
    "phase10_contracts_verified",
    "utility_and_selector_digests_verified",
    "phase10_system_identity_verified",
    "validation_bank_rebuild_verified",
    "test_bank_rebuild_verified",
    "test_bank_structural_audit_pass",
    "pre_agent7_test_outcome_access_zero",
    "outcome_corpus_train_only_verified",
    "candidate_count_6_verified",
    "selection_validation_only_verified",
    "phase9_checkpoint_unchanged_before_eval",
    "gate_a_recomputed",
    "gate_b_recomputed",
    "gate_c_recomputed",
    "gate_d_recomputed",
    "gate_e_recomputed",
    "gate_f_recomputed",
    "gate_g_recomputed",
    "gate_h_recomputed",
    "final_setup_replay_audit_pass",
    "illegal_actions_zero",
    "nonfinite_zero",
    "opponent_hidden_selector_inputs_zero",
    "phase9_checkpoint_unchanged_after_eval",
    "classification_recomputes_from_gate_rows",
    "full_suite_green",
)

#: `full_suite_green` is a claim about the suite that contains this test, so
#: it is checked against the recorded measurement rather than asserted.
SELF_REFERENTIAL_GATE = "full_suite_green"

pytestmark = pytest.mark.skipif(
    not ACCEPTANCE_PATH.exists(), reason="Agent 7 has not produced artifacts yet"
)


@pytest.fixture(scope="module")
def acceptance():
    return json.loads(ACCEPTANCE_PATH.read_text())


@pytest.fixture(scope="module")
def strength_rows():
    with open(STRENGTH_PATH, newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def diversity_rows():
    with open(DIVERSITY_PATH, newline="") as stream:
        return list(csv.DictReader(stream))


def test_the_gate_set_is_exactly_the_instruction(acceptance):
    assert tuple(sorted(acceptance["completion_gates"])) == tuple(sorted(EXPECTED_GATES))


def test_status_follows_from_the_gates_and_the_classification(acceptance):
    non_suite = {
        name: value
        for name, value in acceptance["completion_gates"].items()
        if name != SELF_REFERENTIAL_GATE
    }
    hard_pass = all(acceptance["gates"][name]["pass"] for name in HARD_GATE_IDS)
    if not all(non_suite.values()):
        expected = "BLOCKED"
    elif not hard_pass:
        expected = "FAIL"
    else:
        expected = acceptance["classification"]
    assert acceptance["status"] == expected
    assert acceptance["recommendation"] == expected
    assert acceptance["recommendation"] in (
        "PASS-IMPROVED", "PASS-NONINFERIOR", "FAIL", "BLOCKED"
    )


def test_the_classification_recomputes_from_its_own_gate_rows(acceptance):
    assert classify(acceptance["gates"]) == acceptance["classification"]
    hard_pass = all(acceptance["gates"][name]["pass"] for name in HARD_GATE_IDS)
    assert acceptance["classification_logic"]["hard_gates_all_pass"] == hard_pass
    if acceptance["classification"] == "PASS-IMPROVED":
        assert acceptance["gates"]["A"]["improved"]
        assert acceptance["gates"]["B"]["significantly_positive"]


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    suite = acceptance["suite"]
    expected = suite is not None and suite["returncode"] == 0 and suite["failed"] == 0
    assert acceptance["completion_gates"][SELF_REFERENTIAL_GATE] == expected


def test_gate_a_booleans_recompute_from_observed_values(acceptance):
    gate = acceptance["gates"]["A"]
    assert gate["pass"] == (
        gate["ewr"] >= GATE_A["ordinary"]["ewr_min"]
        and gate["lower_bound"] > GATE_A["ordinary"]["lb_min"]
    )
    assert gate["improved"] == (
        gate["ewr"] >= GATE_A["improved"]["ewr_min"]
        and gate["lower_bound"] > GATE_A["improved"]["lb_min"]
    )


def test_gate_b_booleans_recompute_from_observed_values(acceptance):
    gate = acceptance["gates"]["B"]
    weights = GATE_B["league_weights"]
    recomposed = sum(
        weights[name] * gate["components"][name] for name in weights
    )
    assert abs(recomposed - gate["delta_l"]) < 1e-12
    assert gate["pass"] == (
        gate["delta_l"] >= GATE_B["delta_l_min"]
        and gate["interval"]["lower"] > GATE_B["lb_min"]
    )
    assert gate["significantly_positive"] == (
        gate["delta_l"] > GATE_B["significant"]["delta_l_min"]
        and gate["interval"]["lower"] > GATE_B["significant"]["lb_min"]
    )


def test_gate_e_worst_case_respects_every_frozen_threshold(acceptance):
    gate = acceptance["gates"]["E"]
    observed = gate["observed"]
    thresholds = DIVERSITY_THRESHOLDS
    expected_pass = (
        observed["normalized_family_entropy"] >= thresholds["normalized_family_entropy_min"]
        and observed["effective_families"] >= thresholds["effective_families_min"]
        and observed["family_probability_min"] >= thresholds["family_probability_min"]
        and observed["family_probability_max"] <= thresholds["family_probability_max"]
        and observed["within_family_base_entropy"]
        >= thresholds["within_family_normalized_base_entropy_min"]
        and observed["conditional_base_probability_max"]
        <= thresholds["max_conditional_base_probability"]
    )
    assert gate["pass"] == expected_pass


def test_gate_f_counters_are_all_zero_when_it_passes(acceptance):
    gate = acceptance["gates"]["F"]
    if gate["pass"]:
        assert all(value == 0 for value in gate["counters"].values())
    assert set(gate["counters"]) == {
        "illegal_setups", "inventory_errors", "stranded_sampled_setups",
        "split_leakage", "provenance_mismatch", "hidden_opponent_selector_inputs",
        "illegal_neural_moves", "non_finite_selector_outputs", "inference_failures",
    }


def test_gate_h_names_the_accepted_phase9_identity(acceptance):
    gate = acceptance["gates"]["H"]
    assert gate["expected"]["checkpoint_sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert gate["expected"]["model_state_digest"] == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    if gate["pass"]:
        assert gate["observed"]["checkpoint_sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256


def test_phase9_preservation_holds_before_and_after(acceptance):
    preservation = acceptance["phase9_preservation"]
    assert preservation["before"]["sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert preservation["after"]["sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert preservation["before"]["model_state_digest"] == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert preservation["after"]["model_state_digest"] == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert preservation["unchanged"] is True
    assert preservation["c1_optimizer_steps"] == 0
    assert preservation["parameters"] == 863_959


def test_the_two_system_identities_are_distinguished(acceptance):
    system = acceptance["phase10_system_v1"]
    assert system["frozen_template_digest"] == CONTRACT_DIGESTS["phase10_system_v1"]
    assert system["filled_instance_digest"] == ACCEPTED_SYSTEM_DIGEST
    assert system["frozen_template_digest"] != system["filled_instance_digest"]
    assert system["all_filling_rules_pass"] is True
    assert all(system["filling_rules"].values())


def test_critical_identities_match_the_accepted_freeze(acceptance):
    identities = acceptance["critical_identities"]
    assert identities["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST
    assert identities["validation_bank_digest"] == BANK_DIGESTS["validation"]
    assert identities["test_bank_digest"] == BANK_DIGESTS["test"]
    assert identities["phase9_checkpoint_sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert identities["selector_config_sha256"] == ACCEPTED_CONFIG_SHA256
    assert identities["phase10_system_v1_instance_digest"] == ACCEPTED_SYSTEM_DIGEST
    for name, digest in CONTRACT_DIGESTS.items():
        assert identities["contract_digests"][name] == digest


def test_the_final_bank_and_bootstrap_identity(acceptance):
    bank = acceptance["bank"]
    assert bank["bank_version"] == "phase10_test_bank_v1"
    assert bank["bank_digest"] == BANK_DIGESTS["test"]
    assert bank["cases"] == 512
    assert bank["bootstrap_root"] == TEST_BOOTSTRAP_SEED == 2026081808
    assert bank["bootstrap_replicates"] == 10_000


def test_exactly_one_outcome_bearing_access_exists(acceptance):
    final = acceptance["final_evaluation_access"]
    assert final["count"] == 1
    entry = final["entries"][0]
    assert entry["bank"] == "phase10_test_bank_v1"
    assert entry["purpose"] == "final_evaluation"
    assert entry["stage"] == "games"
    prior = acceptance["pre_agent7_test_bank_access"]
    assert prior["all_entries_structural"] is True
    assert prior["prior_outcome_evaluations"] == 0
    for entry in acceptance["pre_agent7_ledger_entries"]:
        assert entry["neural"] is False
        assert entry["outcomes"] is False


def test_the_evaluated_system_is_the_permanently_selected_one(acceptance):
    system = acceptance["evaluated_system"]
    assert system["candidate_id"] == "P10-D"
    assert system["utility_model"] == "model_T"
    assert system["temperature"] == 0.75
    assert system["selector_config_sha256"] == ACCEPTED_CONFIG_SHA256
    assert system["phase10_system_v1_digest"] == ACCEPTED_SYSTEM_DIGEST
    assert system["baseline"] == "neutral_v1"
    discipline = acceptance["discipline"]
    assert discipline["candidates_evaluated_on_test_bank"] == 1
    assert discipline["utility_models_fit"] == 0
    assert discipline["winner_switches_after_test"] == 0
    assert discipline["report_only_metrics_used_in_gates"] == 0
    assert discipline["c1_optimizer_steps"] == 0


def test_the_landing_diagnostic_is_report_only(acceptance):
    landing = acceptance["landing_diagnostic"]
    assert landing["use"] == "report_only"
    assert landing["gate"] is False
    assert landing["granularity"] == "candidate x arm x matchup x bank"
    assert len(landing["rows"]) == 11
    for row in landing["rows"]:
        assert row["bank"] == "phase10_test_bank_v1"
        assert 0.0 <= row["landing_rate"] <= 1.0


def test_the_games_account_for_both_arms(acceptance):
    games = acceptance["games"]
    assert games["learned_arm"] == 6 * 512 * 2
    assert games["neutral_arm"] == 5 * 512 * 2
    assert games["total"] == games["learned_arm"] + games["neutral_arm"] == 11_264
    assert games["inference"]["failures_returned"] == 0


#: The CSVs quantize to six decimal places by design; the exact values live
#: in the acceptance JSON. Half a unit in the sixth decimal place is the
#: largest disagreement rounding alone can produce.
CSV_ROUNDING = 5e-7


def test_strength_rows_agree_with_the_acceptance_summaries(acceptance, strength_rows):
    assert len(strength_rows) == 11
    matchups = acceptance["matchups"]
    for row in strength_rows:
        summary = matchups[row["matchup"]]
        games = int(row["games"])
        assert games == 1024
        ewr = (int(row["wins"]) + 0.5 * int(row["draws"])) / games
        assert abs(ewr - float(row["ewr"])) < CSV_ROUNDING
        if row["arm"] == "learned":
            assert row["candidate_id"] == "P10-D"
            assert abs(float(row["ewr"]) - summary["learned_ewr"]) < CSV_ROUNDING
        else:
            assert row["candidate_id"] == "neutral_v1"
            assert abs(float(row["ewr"]) - summary["neutral_ewr"]) < CSV_ROUNDING
        if row["delta"]:
            assert abs(float(row["delta"]) - summary["delta"]) < CSV_ROUNDING
            assert (
                abs(float(row["delta_ci_lower"]) - summary["delta_interval"]["lower"])
                < CSV_ROUNDING
            )


def test_diversity_rows_cover_all_36_cells_and_pass(acceptance, diversity_rows):
    assert len(diversity_rows) == 36
    cells = {(row["candidate_id"], row["color"], row["split"]) for row in diversity_rows}
    assert len(cells) == 36
    for row in diversity_rows:
        assert row["mixture_exact"] == "True"
        assert row["all_finite"] == "True"
        assert row["all_thresholds_pass"] == "True"
    worst_entropy = min(float(row["normalized_family_entropy"]) for row in diversity_rows)
    gate_e = acceptance["gates"]["E"]
    assert (
        abs(worst_entropy - gate_e["observed"]["normalized_family_entropy"])
        < CSV_ROUNDING
    )


def test_the_report_carries_the_agent_7_section():
    text = REPORT_PATH.read_text()
    marker = "## 7. Agent 7 — Independent Final Acceptance and Phase 10 Freeze"
    assert marker in text
    section = text[text.index(marker):]
    assert "Recommendation" in section
    assert "report-only" in section.lower()
