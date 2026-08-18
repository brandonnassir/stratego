"""Regression: Agent 2's accepted corpus artifacts stay self-consistent.

Agent 2 produces outcome evidence and nothing else, so these tests pin what
that evidence has to keep saying:

- the corpus is exactly 16,384 games over 256 ordered family pairs at 64
  each, all from the Phase 7 train split;
- the record schema still contains Agent 1's frozen 27 fields as a subset,
  and the fields added beyond them are exactly the ones the acceptance
  artifact declares;
- every upstream identity the artifacts name equals the live value — the
  accepted Phase 9 checkpoint before *and* after collection, the Phase 7
  library, the contract bundle and the outcome schedule;
- the replay audit, the reconstruction audit and the wrong-checkpoint
  negative control all recorded a pass, and the status follows from the
  gates rather than being asserted beside them;
- nothing in the artifacts records a utility fit, a candidate selection, a
  held-out base, or any access to either evaluation bank.

Everything is gated on the artifacts existing, so the suite is green both
before and after Agent 2 runs.
"""

import csv
import json
from pathlib import Path

import pytest

from stratego.training import phase10_contract as pc
from stratego.training import phase10_outcome_store as store
from stratego.training import phase10_schedule as sch
from tests.training.phase10_frozen_digests import (
    CONTRACT_BUNDLE_DIGEST,
    CONTRACT_DIGESTS,
    OUTCOME_SCHEDULE_DIGEST,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_10_data"

ARTIFACTS = {
    "corpus": DATA_DIRECTORY / "agent_02_outcome_corpus.json",
    "acceptance": DATA_DIRECTORY / "agent_02_acceptance.json",
    "family_pairs": DATA_DIRECTORY / "agent_02_family_pair_audit.csv",
}

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in ARTIFACTS.values()),
    reason="Phase 10 Agent 2 artifacts have not been written yet",
)


@pytest.fixture(scope="module")
def corpus():
    return json.loads(ARTIFACTS["corpus"].read_text())


@pytest.fixture(scope="module")
def acceptance():
    return json.loads(ARTIFACTS["acceptance"].read_text())


@pytest.fixture(scope="module")
def family_pairs():
    with ARTIFACTS["family_pairs"].open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status_follows_from_the_recorded_gates(acceptance):
    gates = acceptance["completion_gates"]
    false_gates = sorted(name for name, value in gates.items() if not value)
    assert acceptance["false_gates"] == false_gates
    assert acceptance["gates_total"] == len(gates)
    assert acceptance["gates_true"] == sum(bool(value) for value in gates.values())
    assert acceptance["status"] == ("PASS" if not false_gates else "FAIL")


def test_the_instruction_completion_gates_are_all_present(acceptance):
    required = {
        "agent1_pass",
        "contract_digests_match",
        "phase9_checkpoint_verified_before",
        "phase9_checkpoint_verified_after",
        "phase9_model_state_unchanged",
        "phase7_train_only",
        "games_exact_16384",
        "ordered_pairs_exact_256",
        "games_per_pair_exact_64",
        "duplicate_game_ids_zero",
        "commit_protocol_pass",
        "crash_resume_pass",
        "invalid_setups_zero",
        "stranded_sampled_setups_zero",
        "inventory_violations_zero",
        "setup_provenance_mismatches_zero",
        "illegal_neural_actions_zero",
        "nonfinite_inference_zero",
        "replay_audit_pass",
        "wrong_checkpoint_negative_control_fires",
        "test_bank_neural_outcome_access_zero",
        "no_setup_learning",
        "full_suite_green",
    }
    assert required <= set(acceptance["completion_gates"])


def test_both_artifacts_agree_on_status(corpus, acceptance):
    assert corpus["status"] == acceptance["status"]


# ---------------------------------------------------------------------------
# The corpus itself
# ---------------------------------------------------------------------------


def test_the_corpus_is_the_frozen_schedule(corpus):
    schedule = corpus["schedule"]
    assert schedule["total_games"] == sch.TOTAL_CORPUS_GAMES == 16_384
    assert schedule["ordered_family_pairs"] == sch.ORDERED_FAMILY_PAIRS == 256
    assert schedule["games_per_ordered_pair"] == sch.GAMES_PER_ORDERED_PAIR == 64
    assert schedule["split"] == sch.CORPUS_SPLIT == "train"
    assert schedule["sampler_profile"] == sch.CORPUS_SAMPLER_PROFILE == "neutral_v1"
    assert schedule["schedule_digest"] == OUTCOME_SCHEDULE_DIGEST == sch.schedule_digest()


def test_the_balance_audit_passed_on_the_real_counts(corpus):
    balance = corpus["balance_audit"]
    assert balance["all_pass"]
    assert balance["committed_games"] == 16_384
    assert balance["ordered_pair_count"] == 256
    assert balance["games_per_ordered_pair"] == [64]
    for name in (
        "total_games_exact",
        "ordered_pairs_exact",
        "ordered_pairs_complete",
        "games_per_pair_exact",
        "duplicate_game_ids_zero",
        "duplicate_commit_identities_zero",
        "train_split_violations_zero",
        "setup_provenance_mismatches_zero",
        "policy_identity_mismatches_zero",
    ):
        assert balance["checks"][name], name


def test_one_move_policy_played_every_game(corpus):
    balance = corpus["balance_audit"]
    assert len(balance["move_policy_identities"]) == 1
    assert len(balance["move_model_state_digests"]) == 1
    assert len(balance["move_checkpoint_sha256"]) == 1
    assert balance["move_model_state_digests"][0] == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert balance["move_checkpoint_sha256"][0] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256


def test_the_result_counts_add_up(corpus):
    balance = corpus["balance_audit"]
    assert set(balance["result_counts"]) == {"red_win", "draw", "red_loss"}
    assert sum(balance["result_counts"].values()) == 16_384
    assert sum(balance["result_rates"].values()) == pytest.approx(1.0)


def test_the_corpus_is_sealed_and_its_seal_verifies(corpus):
    assert corpus["seal"]["committed_games"] == 16_384
    assert corpus["seal"]["corpus_version"] == sch.CORPUS_VERSION
    assert corpus["seal_verification"]["all_pass"]
    assert (
        corpus["seal_verification"]["observed_content_digest"] == corpus["seal"]["content_digest"]
    )
    for name in ("state_is_sealed", "content_digest_matches", "committed_games_match"):
        assert corpus["seal_verification"]["checks"][name], name


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_the_frozen_27_field_schema_is_still_a_subset(corpus):
    schema = corpus["record_schema"]
    frozen = {entry["name"] for entry in schema["frozen"]["fields"]}
    assert schema["frozen"]["field_count"] == 27
    assert frozen == set(store.FROZEN_RECORD_FIELDS)
    assert frozen < set(schema["stored_fields"])
    assert set(schema["additional_fields_beyond_frozen"]) == set(schema["stored_fields"]) - frozen


def test_the_stored_schema_matches_the_live_store(corpus):
    schema = corpus["record_schema"]
    assert schema["stored_fields"] == list(store.ASSEMBLED_RECORD_FIELDS)
    assert schema["setup_section_fields"] == list(store.SETUP_SECTION_FIELDS)
    assert schema["outcome_section_fields"] == list(store.OUTCOME_SECTION_FIELDS)
    assert schema["stored_field_count"] == len(store.ASSEMBLED_RECORD_FIELDS)


def test_setup_and_outcome_halves_stay_separated(corpus):
    schema = corpus["record_schema"]
    assert not set(schema["setup_section_fields"]) & set(schema["outcome_section_fields"])


def test_every_added_field_is_declared_as_a_deviation(acceptance, corpus):
    topics = {entry["topic"] for entry in acceptance["deviations"]}
    assert "stored record field count" in topics
    assert corpus["record_schema"]["additional_fields_beyond_frozen"]


# ---------------------------------------------------------------------------
# Upstream identity and preservation
# ---------------------------------------------------------------------------


def test_the_recorded_upstream_identities_are_the_live_ones(acceptance):
    frozen = acceptance["frozen_inputs"]
    assert frozen["phase9_checkpoint_sha256"] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert frozen["phase9_model_state_digest"] == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert frozen["phase9_parameters"] == pc.ACCEPTED_PHASE9_PARAMETERS
    assert frozen["c1_config_digest"] == pc.ACCEPTED_C1_CONFIG_DIGEST
    assert frozen["phase7_library_content_digest"] == pc.PHASE7_LIBRARY_CONTENT_DIGEST
    assert frozen["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST == pc.contract_bundle_digest()
    assert frozen["outcome_schedule_digest"] == OUTCOME_SCHEDULE_DIGEST


def test_the_corpus_contract_digest_is_the_frozen_one(acceptance):
    assert (
        acceptance["new_digests"]["corpus_contract_digest"]
        == CONTRACT_DIGESTS["phase10_setup_outcome_corpus_v1"]
        == pc.document_digest(sch.corpus_contract_document())
    )


def test_phase9_is_byte_identical_before_and_after(acceptance):
    preservation = acceptance["phase9_preservation"]
    assert preservation["before"]["sha256"] == preservation["after"]["sha256"]
    assert preservation["before"]["model_state_digest"] == preservation["after"]["model_state_digest"]
    assert preservation["before"]["parameters"] == preservation["after"]["parameters"]
    assert preservation["file_sha_unchanged"]
    assert preservation["model_state_unchanged"]
    assert preservation["parameters_unchanged"]
    assert preservation["c1_optimizer_steps"] == 0
    assert preservation["after"]["sha256"] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256


def test_the_move_behaviour_is_the_frozen_one(corpus):
    behaviour = corpus["move_policy"]["move_behavior"]
    assert behaviour == dict(sch.CORPUS_MOVE_BEHAVIOR)
    assert behaviour["decision_mode"] == "greedy"
    assert behaviour["dtype"] == "float32"
    assert behaviour["batch_policy"] == "single_request"
    assert behaviour["search"] == "none"
    assert behaviour["optimizer_steps"] == 0
    assert corpus["move_policy"]["phase9_export_bitwise_identical"]


# ---------------------------------------------------------------------------
# Audits and discipline
# ---------------------------------------------------------------------------


def test_the_replay_audit_met_its_minimum_with_no_mismatch(corpus):
    replay = corpus["replay_audit"]
    assert replay["all_pass"]
    assert replay["replayed_games"] >= 2_048
    assert replay["mismatches"] == []
    assert replay["checks"]["outcomes_identical"]
    assert replay["checks"]["all_families_covered"]
    assert replay["families_covered"] == 16


def test_every_stored_setup_rebuilt_from_provenance(corpus):
    reconstruction = corpus["reconstruction_audit"]
    assert reconstruction["all_pass"]
    assert reconstruction["mismatches"] == []
    assert reconstruction["sides_rebuilt"] == 2 * 16_384


def test_the_wrong_checkpoint_negative_control_fired(corpus):
    negative = corpus["negative_control"]
    assert negative["all_pass"]
    assert negative["checks"]["result_verifier_fires"]
    assert negative["games_with_different_outcome"] > 0
    assert negative["wrong_model_state_digest"] != negative["accepted_model_state_digest"]
    assert negative["accepted_model_state_digest"] == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST


def test_no_learning_selection_or_held_out_data_happened(acceptance):
    discipline = acceptance["discipline"]
    assert discipline["utility_models_fit"] == 0
    assert discipline["candidates_selected"] == 0
    assert discipline["c1_optimizer_steps"] == 0
    assert discipline["held_out_bases_in_corpus"] == 0
    assert discipline["validation_bank_outcome_access"] == 0
    assert discipline["test_bank_outcome_access"] == 0
    assert discipline["neural_inference_on_either_bank"] == 0
    assert discipline["human_games_used"] == 0


def test_no_bank_access_played_a_game_or_ran_a_model(acceptance):
    for entry in acceptance["bank_access_log"]:
        assert entry["neural"] is False
        assert entry["outcomes"] is False
        assert entry["purpose"] in {"digest_computation", "structural_audit", "structural_build"}


def test_the_crash_resilience_drill_covered_every_stage(acceptance):
    resilience = acceptance["crash_resilience"]
    assert resilience["all_pass"]
    stages = {entry["stage"] for entry in resilience["crash_drills"]}
    assert {
        "before_payload",
        "after_payload",
        "after_metadata",
        "before_commit_flush",
        "after_commit",
        "shard_rollover",
    } <= stages
    for entry in resilience["crash_drills"]:
        assert entry["pass"], entry["stage"]
        assert entry["crash_fired"]
        assert entry["committed_matches_expected"]
    assert resilience["kill_drill"]["pass"]
    assert resilience["kill_drill"]["killed_exitcode"] != 0
    assert resilience["kill_drill"]["resume_replayed_only_missing"]
    assert resilience["partition_drill"]["pass"]
    assert resilience["partition_drill"]["digests_match"]


# ---------------------------------------------------------------------------
# Storage and handoff
# ---------------------------------------------------------------------------


def test_no_logical_identity_carries_a_physical_path(acceptance, corpus):
    for key, value in acceptance["new_digests"].items():
        assert "/" not in str(value), key
    assert "/" not in corpus["seal"]["content_digest"]
    assert corpus["corpus_version"] == sch.CORPUS_VERSION
    assert "identity_rule" in corpus["storage"]


def test_storage_diagnostics_are_real_measurements(corpus):
    storage = corpus["storage"]
    assert storage["committed_games"] == 16_384
    assert storage["total_bytes"] > 0
    assert storage["bytes_per_game"] > 0
    assert 0.0 < storage["compression_ratio"] < 1.0
    assert storage["free_bytes"] > storage["total_bytes"]


def test_the_handoff_tells_agent_3_where_the_corpus_is_and_what_it_may_use(acceptance):
    handoff = acceptance["handoff_to_agent_3"]
    assert handoff["for_agent"] == 3
    assert handoff["corpus_state"] == store.STATE_SEALED
    assert handoff["corpus_version"] == sch.CORPUS_VERSION
    assert handoff["corpus_content_digest"]
    assert handoff["standardization_source"]["train_bases"] == 6_400
    assert handoff["standardization_source"]["held_out_bases_in_corpus"] == 0
    assert handoff["proof_no_leak"]["test_bank_outcome_access"] == 0
    assert handoff["proof_no_leak"]["c1_optimizer_steps"] == 0
    assert handoff["schema"]["result_targets"] == {"red_win": 1.0, "draw": 0.5, "red_loss": 0.0}
    assert set(handoff["schema"]["stored_fields"]) == set(store.ASSEMBLED_RECORD_FIELDS)


# ---------------------------------------------------------------------------
# The CSV
# ---------------------------------------------------------------------------


def test_the_family_pair_csv_covers_all_256_ordered_pairs(family_pairs):
    assert len(family_pairs) == 256
    pairs = {(row["red_family"], row["blue_family"]) for row in family_pairs}
    assert pairs == set(sch.ordered_family_pairs())


def test_every_family_pair_row_holds_exactly_64_games(family_pairs):
    total = 0
    for row in family_pairs:
        games = int(row["games"])
        assert games == 64
        assert int(row["red_wins"]) + int(row["draws"]) + int(row["red_losses"]) == games
        assert 0.0 <= float(row["red_score"]) <= 1.0
        assert float(row["mean_plies"]) > 0
        total += games
    assert total == 16_384


def test_the_csv_shows_setup_diversity_inside_every_pair(family_pairs):
    for row in family_pairs:
        assert int(row["distinct_red_bases"]) > 1
        assert int(row["distinct_blue_bases"]) > 1


def test_the_suite_recorded_green(acceptance):
    assert acceptance["suite"]["failed"] == 0
    assert acceptance["suite"]["returncode"] == 0
    assert acceptance["suite"]["passed"] > acceptance["suite_before"]["passed"]
