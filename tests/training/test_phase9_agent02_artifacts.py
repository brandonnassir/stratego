"""Regression: Agent 2's schedule artifacts stay true to live source.

Agent 2 publishes the logical Phase 9 schedule, its exhaustive audits, and
the storage handoff Agent 3 will act on. These tests pin what that publication
*means*:

- the four artifacts exist together and carry a status justified by their own
  recorded gates;
- every recorded digest equals the live `phase9_schedule` value, so an edit to
  the schedule arithmetic cannot leave a stale artifact standing;
- the recorded totals really are 6 x 8 x 1,024 pilot games and 60 x 2,048
  canonical games with no duplicate or cross-namespace identity;
- the audits recorded zero seed collisions, zero split violations and zero
  held-out setup leaks;
- the storage handoff separates the logical identity from the resolved path,
  measures free space, and names the resolution source — and no schedule
  digest depends on any of it;
- the CSV summarises exactly the canonical run's 122,880 games.

Everything is gated on the artifacts existing, so the suite is green both
before and after Agent 2 runs.
"""

import csv
import json
from pathlib import Path

import pytest

from stratego.training import phase9_contract as pc
from stratego.training import phase9_schedule as psch
from stratego.training import phase9_seed as pseed
from stratego.training import phase9_storage as pstore

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_9_data"

ARTIFACTS = {
    "population": DATA_DIRECTORY / "agent_02_population.json",
    "audit": DATA_DIRECTORY / "agent_02_schedule_audit.json",
    "acceptance": DATA_DIRECTORY / "agent_02_acceptance.json",
}
SUMMARY_CSV = DATA_DIRECTORY / "agent_02_canonical_schedule_summary.csv"

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in ARTIFACTS.values()) or not SUMMARY_CSV.exists(),
    reason="Agent 2's schedule artifacts have not been produced yet",
)


@pytest.fixture(scope="module")
def artifacts() -> dict:
    return {name: json.loads(path.read_text()) for name, path in ARTIFACTS.items()}


@pytest.fixture(scope="module")
def summary_rows() -> list:
    with open(SUMMARY_CSV, newline="") as handle:
        return list(csv.DictReader(handle))


class TestArtifactCoherence:
    def test_every_artifact_names_phase_9_agent_2(self, artifacts):
        for name, payload in artifacts.items():
            assert payload["phase"] == 9, name
            assert payload["agent"] == 2, name

    def test_status_is_justified_by_the_recorded_gates(self, artifacts):
        acceptance = artifacts["acceptance"]
        gates = acceptance["completion_gates"]
        assert acceptance["gates_total"] == len(gates)
        assert acceptance["gates_true"] == sum(bool(value) for value in gates.values())
        if acceptance["status"] == "PASS":
            assert all(gates.values())
            assert not acceptance["problems"]

    def test_every_gate_the_assignment_requires_is_present(self, artifacts):
        required = {
            "agent1_pass",
            "contract_digests_match",
            "corpus_resolver_verified",
            "corpus_digests_match",
            "pilot_schedules_exact",
            "canonical_60_iteration_schedule_exact",
            "canonical_total_games_122880",
            "duplicate_game_ids_zero",
            "seed_collision_violations_zero",
            "bucket_count_mismatches_zero",
            "rule_subdivision_mismatches_zero",
            "stress_allocation_mismatches_zero",
            "color_balance_violations_zero",
            "train_setup_split_violations_zero",
            "worker_order_dependence_zero",
            "resume_identity_mismatches_zero",
            "no_neural_training",
            "full_suite_green",
        }
        assert required <= set(artifacts["acceptance"]["completion_gates"])

    def test_the_accepted_agent_1_identities_are_recorded_unchanged(self, artifacts):
        verification = artifacts["acceptance"]["agent1_verification"]
        assert verification["agent1_status"] == "PASS"
        assert verification["contract_digest"] == pc.contract_digest()
        assert verification["contract_digest_matches_accepted"]
        assert verification["bank_digests_match_accepted"]
        assert verification["canonical_seeds"] == {
            name: int(value) for name, value in pseed.CANONICAL_PHASE9_SEEDS.items()
        }

    def test_the_phase_8_corpus_was_resolved_and_matched(self, artifacts):
        corpus = artifacts["acceptance"]["corpus_verification"]
        assert corpus["resolver"].endswith("default_corpus_root()")
        assert corpus["resolved_root_matches_accepted_location"]
        assert corpus["identity_matches"]
        assert corpus["observed_identity"] == corpus["accepted_identity"]
        assert corpus["scheduler_hardcodes_corpus_path"] is False


class TestDigestsMatchLiveSource:
    def test_population_digest_matches_live_source(self, artifacts):
        live = psch.population_digest()
        assert artifacts["population"]["population_digest"] == live
        assert artifacts["audit"]["population_digest"] == live
        assert artifacts["acceptance"]["population_digest"] == live

    def test_population_document_matches_live_source(self, artifacts):
        assert artifacts["population"]["population"] == json.loads(
            json.dumps(psch.population_document())
        )

    @pytest.mark.parametrize("namespace", pseed.RUN_NAMESPACES)
    def test_run_schedule_digests_match_live_source(self, artifacts, namespace):
        recorded = artifacts["acceptance"]["run_schedule_digests"][namespace]
        assert recorded == psch.run_schedule_digest(namespace)
        assert artifacts["audit"]["run_schedule_digests"][namespace] == recorded

    def test_recorded_scheduled_games_rebuild_from_their_own_ids(self, artifacts):
        for example in artifacts["population"]["scheduled_game_examples"]:
            rebuilt = psch.rebuild_scheduled_game(example["phase9_game_id"])
            assert rebuilt.to_dict() == example

    def test_record_fields_cover_the_assignment_game_record(self, artifacts):
        fields = set(artifacts["population"]["record_fields"])
        assert {
            "phase9_game_id",
            "run_namespace",
            "rl_iteration",
            "game_ordinal",
            "bucket",
            "red_policy_identity",
            "blue_policy_identity",
            "learner_control",
            "behavior_snapshot_identity",
            "historical_snapshot_identity",
            "setup_root_seed",
            "red_setup_source_identity",
            "blue_setup_source_identity",
        } <= fields


class TestRecordedAudits:
    def test_recorded_totals_are_the_frozen_schedule(self, artifacts):
        totals = artifacts["acceptance"]["schedule_totals"]
        assert totals["canonical_iterations"] == 60
        assert totals["canonical_games_per_iteration"] == 2048
        assert totals["canonical_total_games"] == 122_880
        assert totals["pilot_runs"] == 6
        assert totals["pilot_iterations_each"] == 8
        assert totals["pilot_games_per_iteration"] == 1024
        assert totals["pilot_total_games_each"] == 8192
        assert totals["all_namespaces_total_games"] == 172_032
        assert totals["distinct_game_ids"] == totals["all_namespaces_total_games"]

    def test_no_seed_collisions_were_recorded(self, artifacts):
        seeds = artifacts["acceptance"]["seed_collisions"]
        assert seeds["within_stream_collisions"] == 0
        assert seeds["same_game_setup_side_collisions"] == 0
        assert all(count == 0 for count in seeds["per_stream"].values())
        assert seeds["seeds_derived"] > 172_032

    def test_the_audited_seed_streams_are_the_ones_the_contract_names(self, artifacts):
        assert set(artifacts["acceptance"]["seed_collisions"]["per_stream"]) == {
            "setup_root",
            "setup_side_red",
            "setup_side_blue",
            "policy_red",
            "policy_blue",
            "historical_opponent",
        }

    def test_setups_were_train_split_only_with_full_family_coverage(self, artifacts):
        setups = artifacts["acceptance"]["setup_assignment"]
        assert setups["split"] == "train"
        assert setups["purpose"] == "training"
        assert setups["profile"] == "neutral_v1"
        assert setups["split_violations"] == 0
        assert setups["families_seen"] == 16
        assert setups["held_out_setup_leaks"] == 0
        assert setups["held_out_setups_compared"] > 0

    def test_the_setup_audit_ran_no_model_and_no_engine(self, artifacts):
        setups = artifacts["acceptance"]["setup_assignment"]
        assert setups["models_constructed"] == 0
        assert setups["checkpoints_loaded"] == 0
        assert setups["engine_plies_simulated"] == 0

    def test_per_iteration_audits_recorded_no_problems(self, artifacts):
        for namespace, report in artifacts["audit"]["namespaces"].items():
            assert not report["problems"], namespace
            for iteration in report["per_iteration"]:
                assert not iteration["problems"], (namespace, iteration["iteration"])
                assert (
                    iteration["bucket_counts"] == iteration["expected_bucket_counts"]
                )
                assert (
                    iteration["rule_tier_counts"]
                    == iteration["expected_rule_tier_counts"]
                )
                assert iteration["stress_spread"] <= 1

    def test_worker_order_and_resume_audits_recorded_no_mismatch(self, artifacts):
        for key, report in artifacts["audit"]["worker_order_independence"].items():
            assert report["mismatches"] == 0, key
            assert report["partitionings_checked"] >= 12, key
        for key, report in artifacts["audit"]["resume_identity"].items():
            assert not report["problems"], key
            assert report["foreign_committed_id_rejected"], key


class TestStorageHandoff:
    def test_the_handoff_separates_identity_from_location(self, artifacts):
        handoff = artifacts["acceptance"]["storage_handoff"]
        identity = handoff["logical_schedule_identity"]
        assert identity["population_version"] == pc.PHASE9_POPULATION_VERSION
        assert identity["schedule_version"] == pc.PHASE9_ROLLOUT_SCHEDULE_VERSION
        assert identity["population_digest"] == psch.population_digest()
        assert set(identity["run_schedule_digests"]) == set(pseed.RUN_NAMESPACES)
        # The identity block must contain no path at all.
        assert "/" not in json.dumps(identity)

    def test_the_handoff_records_the_resolved_root_and_its_source(self, artifacts):
        handoff = artifacts["acceptance"]["storage_handoff"]
        assert handoff["resolved_rollout_root"]
        assert handoff["storage_resolution_source"]
        assert handoff["resolver"].endswith("default_rollout_root()")
        assert handoff["rollout_corpus_created"] is False

    def test_the_handoff_reports_a_measured_free_space(self, artifacts):
        measurement = artifacts["acceptance"]["storage_handoff"][
            "free_space_measurement"
        ]
        assert measurement["free_bytes"] > 0
        assert measurement["total_bytes"] >= measurement["free_bytes"]
        assert (
            measurement["free_bytes"]
            > measurement["projected_requirement_bytes"] * pstore.REQUIRED_HEADROOM_FACTOR
        )
        assert measurement["observed_headroom_factor"] >= pstore.REQUIRED_HEADROOM_FACTOR

    def test_the_handoff_states_that_the_path_is_not_identity(self, artifacts):
        handoff = artifacts["acceptance"]["storage_handoff"]
        assert handoff["path_is_diagnostic_not_identity"] == pstore.STORAGE_IDENTITY_RULE

    def test_no_recorded_digest_contains_the_storage_root(self, artifacts):
        root = artifacts["acceptance"]["storage_handoff"]["resolved_rollout_root"]
        assert root not in json.dumps(artifacts["population"]["population"])
        assert root not in json.dumps(
            artifacts["acceptance"]["storage_handoff"]["logical_schedule_identity"]
        )

    def test_agent_3_is_warned_about_per_side_checkpoint_identity(self, artifacts):
        warning = artifacts["acceptance"]["handoff_to_agent_3"][
            "per_side_checkpoint_identity_warning"
        ]
        assert "collection_checkpoint_id" in warning
        assert "historical" in warning

    def test_the_pilot_archive_consequence_is_recorded(self, artifacts):
        # The frozen window rule applies in every namespace, so an 8-iteration
        # pilot really does schedule H005 opponents. Agents 5/6 have to archive
        # a pilot snapshot after iteration 5 for those games to be playable,
        # and that consequence must not be discovered at run time.
        notes = " ".join(artifacts["acceptance"]["carry_forward_notes"])
        assert "H005" in notes
        assert "pilot iteration 5" in notes
        pilot = artifacts["audit"]["namespaces"]["pilot_p9a"]["historical_totals"]
        assert pilot["H005"] > 0
        assert set(pilot) == {"H000", "H005"}

    def test_the_handoff_names_the_apis_agent_3_needs(self, artifacts):
        handoff = artifacts["acceptance"]["handoff_to_agent_3"]
        for key in (
            "schedule_enumeration_api",
            "game_id_parser_rebuilder",
            "active_history_manifest_interface",
            "learner_control_field",
            "setup_identity_derivation",
            "resume_subtraction",
        ):
            assert handoff[key], key
        assert handoff["no_new_learning_design_decisions"] is True


class TestCanonicalSummaryCsv:
    def test_the_summary_covers_the_whole_canonical_run(self, summary_rows):
        assert sum(int(row["games"]) for row in summary_rows) == 122_880
        assert {int(row["rl_iteration"]) for row in summary_rows} == set(range(1, 61))
        assert {row["namespace"] for row in summary_rows} == {"canonical"}

    def test_the_summary_has_the_assignment_columns(self, summary_rows):
        assert set(summary_rows[0]) == {
            "namespace",
            "rl_iteration",
            "bucket",
            "opponent_kind",
            "opponent_identity",
            "learner_control",
            "learner_color",
            "games",
        }

    def test_each_iteration_sums_to_2048_with_the_frozen_bucket_split(
        self, summary_rows
    ):
        per_iteration: dict = {}
        for row in summary_rows:
            key = (int(row["rl_iteration"]), row["bucket"])
            per_iteration[key] = per_iteration.get(key, 0) + int(row["games"])
        for iteration in range(1, 61):
            assert per_iteration[(iteration, "current")] == 1024
            assert per_iteration[(iteration, "historical")] == 512
            assert per_iteration[(iteration, "rule")] == 307
            assert per_iteration[(iteration, "stress")] == 205

    def test_self_play_rows_carry_no_learner_colour(self, summary_rows):
        for row in summary_rows:
            if row["bucket"] == "current":
                assert row["learner_control"] == "both"
                assert row["learner_color"] == ""
            else:
                assert row["learner_control"] in ("red", "blue")
                assert row["learner_color"] == row["learner_control"]

    def test_the_summary_agrees_with_the_recorded_audit(self, summary_rows, artifacts):
        recorded = artifacts["audit"]["canonical_summary_csv"]
        assert recorded["rows"] == len(summary_rows)
        assert recorded["games"] == sum(int(row["games"]) for row in summary_rows)
