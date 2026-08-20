"""Phase 11 Agent 3 artifacts: the sampler audit checks itself.

Every check recomputes something from the tracked artifacts and the live
modules rather than trusting a stored summary. `full_suite_green` is a claim
about the suite that contains this test, so it is checked against the
recorded measurement rather than asserted (the accepted Phase 10/Agent 2
pattern).
"""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from stratego.training import phase11_contract as pc
from stratego.training.phase11_seed import OPPONENT_STRATA

from ..training.phase11_frozen_digests import (
    BANK_DIGESTS,
    BELIEF_HEAD_DIGEST,
    CONTRACT_BUNDLE_DIGEST,
    CONTRACT_DIGESTS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"
ACCEPTANCE_PATH = DATA_DIRECTORY / "agent_03_acceptance.json"
CONTRACT_PATH = DATA_DIRECTORY / "agent_03_sampler_contract.json"
AUDIT_PATH = DATA_DIRECTORY / "agent_03_sampler_audit.json"
DIAGNOSTICS_PATH = DATA_DIRECTORY / "agent_03_sampler_diagnostics.csv"

#: The instruction's twenty-four minimum gates plus the two Agent 3
#: additions (collision exhaustiveness, guard public-input proof). Gates
#: may be added, never weakened.
EXPECTED_GATES = (
    "agents1_2_pass",
    "sampler_contract_verified",
    "sampler_request_boundary_exact",
    "true_hidden_inputs_rejected",
    "exact_inventory_enforced",
    "public_masks_enforced",
    "known_ranks_locked",
    "piece_order_seeded",
    "categorical_draw_seeded",
    "zero_mass_fallback_exact",
    "complete_world_validation_exact",
    "sampler_worlds_ge_250k",
    "thousands_distinct_states",
    "all_8_strata_covered",
    "both_colors_covered",
    "all_zero_tolerance_counters_zero",
    "independent_audit_pass",
    "negative_controls_fire",
    "deterministic_repeat_pass",
    "baseline_world_sampler_valid",
    "no_test_prediction_access",
    "no_belief_updates",
    "upstream_artifacts_unchanged",
    "full_suite_green",
    "world_stream_collisions_zero",
    "feasibility_guard_public_inputs_only",
)

SELF_REFERENTIAL_GATE = "full_suite_green"

NEGATIVE_CONTROLS = (
    "remove_one_remaining_rank",
    "bomb_or_flag_on_moved_piece",
    "duplicate_marshal_count",
    "alter_public_known_rank",
    "mutate_sample_seed",
    "inject_true_hidden_rank",
    "corrupt_provenance",
)

pytestmark = pytest.mark.skipif(
    not ACCEPTANCE_PATH.exists(), reason="Agent 3 has not produced artifacts yet"
)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


@pytest.fixture(scope="module")
def acceptance():
    return json.loads(ACCEPTANCE_PATH.read_text())


@pytest.fixture(scope="module")
def contract():
    return json.loads(CONTRACT_PATH.read_text())


@pytest.fixture(scope="module")
def audit():
    return json.loads(AUDIT_PATH.read_text())


# ---------------------------------------------------------------------------
# Gates and status
# ---------------------------------------------------------------------------


def test_the_gate_set_is_the_instruction_list_plus_the_two_additions(acceptance):
    assert tuple(sorted(acceptance["completion_gates"])) == tuple(sorted(EXPECTED_GATES))
    assert acceptance["gates_total"] == len(EXPECTED_GATES)


def test_status_follows_from_the_gates(acceptance):
    gates = acceptance["completion_gates"]
    expected = "PASS" if all(gates.values()) else "BLOCKED"
    assert acceptance["status"] == expected
    assert acceptance["gates_true"] == sum(1 for value in gates.values() if value)
    assert acceptance["false_gates"] == sorted(
        name for name, value in gates.items() if not value
    )


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    suite = acceptance["suite"]
    expected = bool(suite) and suite.get("returncode") == 0
    assert acceptance["completion_gates"][SELF_REFERENTIAL_GATE] == expected


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------


def test_the_frozen_inputs_are_the_agent_1_freeze(acceptance):
    frozen = acceptance["frozen_inputs"]
    assert frozen["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST
    assert frozen["contract_digests"] == CONTRACT_DIGESTS
    assert frozen["sampler_contract_digest"] == CONTRACT_DIGESTS["phase11_belief_sampler_v1"]
    assert frozen["validation_bank_digest"] == BANK_DIGESTS["validation"]
    assert frozen["test_bank_digest"] == BANK_DIGESTS["test"]
    assert frozen["belief_head_digest"] == BELIEF_HEAD_DIGEST
    assert frozen["phase9_sha256"] == pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert frozen["phase9_model_state_digest"] == pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert frozen["phase9_parameters"] == 863_959
    assert frozen["phase10_closure_commit"] == pc.PHASE10_CLOSURE_COMMIT


def test_the_sampler_contract_digest_still_recomputes_live(contract):
    assert contract["contract_digest"] == CONTRACT_DIGESTS["phase11_belief_sampler_v1"]
    assert pc.contract_digests()["phase11_belief_sampler_v1"] == contract["contract_digest"]
    assert contract["frozen_document"] == json.loads(
        json.dumps(pc.sampler_document())
    )


def test_the_implementation_bytes_match_the_recorded_identity(contract, acceptance):
    recorded = contract["implementation_identity"]["module_sha256"]
    assert recorded == acceptance["new_digests"]["implementation_sha256"]
    for module, digest in recorded.items():
        assert _file_sha256(REPOSITORY_ROOT / module) == digest, module


def test_the_audit_artifact_bytes_match_the_recorded_digest(acceptance):
    assert acceptance["new_digests"]["sampler_audit_artifact_sha256"] == _file_sha256(
        AUDIT_PATH
    )
    assert acceptance["new_digests"]["sampler_diagnostics_sha256"] == _file_sha256(
        DIAGNOSTICS_PATH
    )


# ---------------------------------------------------------------------------
# The audit's hard facts
# ---------------------------------------------------------------------------


def test_every_zero_tolerance_counter_is_present_and_zero(audit):
    counters = audit["zero_tolerance_counters"]
    assert sorted(counters) == sorted(pc.SAMPLER_ZERO_TOLERANCE_COUNTERS)
    assert all(value == 0 for value in counters.values()), counters
    assert audit["all_zero_tolerance_counters_zero"]


def test_the_audit_volumes_clear_the_frozen_floors(audit):
    volumes = audit["volumes"]
    assert volumes["learned_worlds"] >= pc.SAMPLER_AUDIT_MIN_WORLDS
    assert volumes["learned_worlds_validated"] == volumes["learned_worlds"]
    assert volumes["independent_worlds"] >= pc.SAMPLER_INDEPENDENT_AUDIT_MIN_WORLDS
    assert volumes["distinct_public_state_identities"] >= 2_000
    assert not audit["smoke_run"]


def test_the_coverage_spans_the_frozen_slices(audit):
    coverage = audit["coverage"]
    assert sorted(coverage["states_by_stratum"]) == sorted(OPPONENT_STRATA)
    assert sorted(coverage["states_by_observer_color"]) == ["blue", "red"]
    assert sorted(coverage["states_by_progress_bucket"]) == ["early", "late", "middle"]
    assert coverage["states_with_moved_uncertainty"] > 0
    assert coverage["states_with_unmoved_uncertainty"] > 0


def test_the_store_integrity_checks_found_nothing(audit):
    integrity = audit["store_integrity"]
    assert integrity["shards_verified"] == 1_024
    for name, value in integrity.items():
        if name.endswith("_mismatches"):
            assert value == 0, name


def test_the_independent_path_agreed_everywhere(audit):
    independent = audit["independent_audit"]
    assert independent["disagreements"] == 0
    assert independent["knife_edge_events"] == 0
    assert independent["worlds"] >= pc.SAMPLER_INDEPENDENT_AUDIT_MIN_WORLDS
    assert independent["steps_recomputed"] > 0
    assert independent["pass"]


def test_the_repeats_were_bit_identical(audit):
    determinism = audit["determinism"]
    assert determinism["repeat_states"] > 0
    assert determinism["repeat_mismatches"] == 0
    assert determinism["reversal_mismatches"] == 0


def test_every_negative_control_fires(audit):
    fired = audit["negative_controls"]["fired"]
    assert tuple(sorted(fired)) == tuple(sorted(NEGATIVE_CONTROLS))
    assert all(fired.values()), fired
    assert audit["negative_controls"]["all_fire"]


def test_the_boundary_probes_all_rejected(audit):
    boundary = audit["boundary"]
    assert boundary["all_probes_rejected"]
    assert boundary["hidden_input_accesses"] == 0
    assert boundary["allowed_request_fields"] == list(pc.ALLOWED_SAMPLER_REQUEST_FIELDS)
    assert all(
        probe["rejected"] for probe in boundary["rejected_input_probes"].values()
    )


def test_the_collision_audit_covers_the_world_streams_exhaustively(audit):
    collision = audit["seed_collision_audit"]
    assert collision["no_collisions"]
    assert collision["findings"] == []
    streams = collision["streams"]
    for name in ("world_sample", "world_order", "world_categorical"):
        assert name in streams
        assert streams[name]["count"] == streams[name]["distinct"]
    # The world_sample stream carries one seed per materialized token:
    # learned worlds plus the baseline tokens of every state.
    volumes = audit["volumes"]
    expected_tokens = volumes["learned_worlds"] + volumes["baseline_worlds"]
    assert streams["world_sample"]["count"] == expected_tokens
    # Order/categorical streams: one seed per (token, slot/step) pair.
    assert streams["world_order"]["count"] == streams["world_categorical"]["count"]
    # The Agent 1 enumerable universe rides along, so the audit is against
    # the whole relevant seed space.
    for name in ("bank_match", "safety_state_selection", "bootstrap"):
        assert name in streams


def test_the_baseline_sampler_was_verified_without_ranking(audit):
    baseline = audit["baseline_sampler"]
    assert baseline["sampler_version"] == pc.WORLD_BASELINE_VERSION
    assert baseline["worlds"] > 0
    assert baseline["all_counters_zero"], baseline["counters"]
    assert baseline["strength_comparison"] == "none, by contract"


def test_the_diagnostics_csv_matches_the_audit(audit):
    with open(DIAGNOSTICS_PATH, newline="") as stream:
        rows = list(csv.DictReader(stream))
    volumes = audit["volumes"]
    assert len(rows) == volumes["states"]
    assert sum(int(row["learned_worlds"]) for row in rows) == volumes["learned_worlds"]
    assert sum(int(row["baseline_worlds"]) for row in rows) == volumes["baseline_worlds"]
    identities = {row["public_state_identity"] for row in rows}
    assert len(identities) == volumes["distinct_public_state_identities"]
    strata = {row["opponent_stratum"] for row in rows}
    assert strata == set(OPPONENT_STRATA)
    assert sum(int(row["fallback_steps"]) for row in rows) == audit["fallback"][
        "fallback_steps_total"
    ]


# ---------------------------------------------------------------------------
# Preservation, counters, ledger, handoff
# ---------------------------------------------------------------------------


def test_nothing_upstream_moved(acceptance):
    preservation = acceptance["preservation"]
    assert preservation["checkpoint_unchanged"]
    assert preservation["belief_head_unchanged"]
    assert preservation["p10d_unchanged"]
    assert preservation["phase7_unchanged"]
    assert preservation["anchor_unchanged"]
    assert preservation["prediction_store_unchanged"]
    assert preservation["bank_files_unchanged"]
    assert preservation["problems"] == []
    assert preservation["optimizer_step_delta"] == 0
    assert preservation["optimizer_steps_run"] == 0


def test_every_forbidden_counter_is_zero(acceptance):
    counters = acceptance["forbidden_operation_counters"]
    assert set(counters) >= {
        "phase11_optimizer_steps",
        "belief_calibration_operations",
        "sampler_weighting_changes",
        "feasibility_rule_changes",
        "test_bank_scored_accesses",
        "validation_truth_shard_reads",
        "hidden_truth_inputs_to_sampling",
    }
    assert all(value == 0 for value in counters.values()), counters


def test_the_test_bank_seal_held_until_agent_7():
    """Agent 3's seal invariant in its permanent time-scoped form: every
    ledger entry up to and including Agent 6 is structural-only with zero
    counters, and the only non-structural test-bank entries are Agent 7's
    single authorized sealed evaluation, which postdates this agent."""
    from stratego.evaluation import phase11_banks as pb

    entries = pb.read_ledger()
    sealing = pb.verify_test_bank_sealed(
        [entry for entry in entries if entry["agent"] <= 6]
    )
    assert sealing["test_bank_structural_only"]
    assert sealing["scored_prediction_total"] == 0
    assert sealing["privileged_truth_total"] == 0
    assert sealing["neural_inference_total"] == 0
    assert sealing["outcome_total"] == 0
    scored = [
        entry
        for entry in entries
        if entry["bank_version"] == "phase11_test_bank_v1"
        and not entry["structural_only"]
    ]
    assert all(entry["agent"] == 7 for entry in scored)


def test_the_ledger_records_only_structural_agent_3_access():
    from stratego.evaluation import phase11_banks as pb

    entries = [entry for entry in pb.read_ledger() if entry["agent"] == 3]
    assert entries, "Agent 3 wrote no ledger entry"
    for entry in entries:
        assert entry["structural_only"], entry
        assert entry["neural_inference_count"] == 0
        assert entry["scored_prediction_count"] == 0
        assert entry["privileged_truth_count"] == 0
        assert entry["outcome_count"] == 0


def test_the_handoff_names_what_agent_4_needs(acceptance):
    handoff = acceptance["handoff_to_agent_4"]
    assert handoff["for_agent"] == 4
    assert handoff["sampler"]["sampler_version"] == pc.BELIEF_SAMPLER_VERSION
    assert "must not change the sampler mathematics" in handoff["sampler"]["immutable"]
    assert handoff["provenance_schema"]["fields"] == list(pc.SAMPLER_PROVENANCE_FIELDS)
    assert handoff["validation_public_states"]["store_pointer"] == (
        "data/phase11_prediction_root.txt"
    )
    assert "0..63" in handoff["sample_id_rules"]["production_ordinals"]


def test_the_readings_are_recorded_for_the_reviewer(acceptance):
    readings = acceptance["recorded_readings"]
    assert readings
    names = {reading["reading"] for reading in readings}
    assert "gate_a_risk_acknowledged_nothing_retuned" in names
    assert "audit_schedule_frozen_before_sampling" in names
    assert "sampler_audit_replays_are_structural" in names
    for reading in readings:
        assert reading["statement"] and reading["impact"]


def test_the_acceptance_summary_mirrors_the_audit_artifact(acceptance, audit):
    summary = acceptance["audit_summary"]
    assert summary["learned_worlds"] == audit["volumes"]["learned_worlds"]
    assert summary["states"] == audit["volumes"]["states"]
    assert summary["zero_tolerance_counters"] == audit["zero_tolerance_counters"]
    assert summary["negative_controls"] == audit["negative_controls"]["fired"]
    assert (
        summary["seed_collision_audit"]["no_collisions"]
        == audit["seed_collision_audit"]["no_collisions"]
    )
