"""Phase 11 Agent 6: the four acceptance artifacts.

These tests read the artifacts the Agent 6 harness wrote and check that
they say what Agent 7 will rely on: the soak really ran the frozen
production shape on non-bank train-only states, the store holds exactly the
scheduled request ids after a real process death, every committed request
was independently re-derived with all zero-tolerance counters at zero,
`phase11_system_v1` is filled by Agent 1's own rules and carries no
absolute path, every preserved identity is exact, and the sealed test bank
was still never scored.

`full_suite_green` is a claim about the suite that contains this test, so
it is checked for consistency against the recorded measurement rather than
asserted — the accepted Phase 10 / Agent 2-5 pattern. The artifacts are
skipped when absent so a fresh clone still runs green.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stratego.training import phase11_contract as contract
from stratego.training.phase11_seed import SOAK_GAME_COUNT, SOAK_REQUEST_COUNT

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_11_data"


def _load(name: str):
    path = DATA_DIRECTORY / name
    if not path.exists():
        pytest.skip(f"{name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def acceptance():
    return _load("agent_06_acceptance.json")


@pytest.fixture(scope="module")
def manifest():
    return _load("agent_06_soak_manifest.json")


@pytest.fixture(scope="module")
def audit():
    return _load("agent_06_soak_audit.json")


@pytest.fixture(scope="module")
def system():
    return _load("agent_06_system_v1.json")


# ---------------------------------------------------------------------------
# The soak actually ran the frozen production shape
# ---------------------------------------------------------------------------


def test_soak_ran_the_frozen_volume(manifest):
    block = manifest["soak"]
    assert block["soak_version"] == "phase11_soak_v1"
    assert block["games"] >= SOAK_GAME_COUNT
    assert block["requests"] >= SOAK_REQUEST_COUNT
    assert block["worlds_per_request"] == 64
    assert block["worlds"] == block["requests"] * 64


def test_every_playable_game_gave_its_full_eight_requests(manifest):
    findings = manifest["soak"]["schedule_findings"]
    assert findings["frozen_request_count"] == SOAK_REQUEST_COUNT == 8192
    assert findings["realizable_request_count"] == manifest["soak"]["requests"]
    assert findings["every_playable_game_gave_eight"] is True
    assert len(findings["zero_decision_game_ids"]) == findings["zero_decision_games"]


def test_the_original_range_is_preserved_exactly_by_the_supplement(manifest):
    """The reviewer's first condition: preserve all 1,024 original games and
    all 7,960 original requests exactly. The supplement only appends."""
    findings = manifest["soak"]["schedule_findings"]
    assert findings["original"]["games"] == SOAK_GAME_COUNT
    assert findings["original_requests_preserved_exactly"] is True
    assert findings["first_difference"] is None
    assert findings["combined_requests"] == (
        findings["original"]["requests"] + findings["supplemental_requests"]
    )


def test_the_supplement_is_exactly_the_authorized_size(manifest):
    supplement = manifest["soak"]["schedule_findings"]["supplement"]
    findings = manifest["soak"]["schedule_findings"]
    if not supplement.get("authorized"):
        pytest.skip("no supplement was needed")
    assert supplement["playable"] == 29
    assert supplement["unplayable"] + supplement["playable"] == (
        supplement["games_enumerated"]
    )
    assert findings["supplemental_requests"] == 29 * 8 == 232
    assert findings["combined_requests"] == SOAK_REQUEST_COUNT
    assert supplement["first_ordinal"] == 128


def test_the_supplement_uses_agent_1s_rules_unchanged(manifest):
    supplement = manifest["soak"]["schedule_findings"]["supplement"]
    if not supplement.get("authorized"):
        pytest.skip("no supplement was needed")
    proof = supplement["rules_proof"]
    assert proof["rules_identical"] is True
    assert proof["mismatches"] == 0
    assert proof["frozen_games_covered"] == SOAK_GAME_COUNT
    assert proof["comparisons"] > 10_000
    assert supplement["enumeration"] == "ordinal-major over the frozen stratum order"


def test_the_supplement_is_carried_as_a_recorded_reading(acceptance):
    readings = {row["reading"] for row in acceptance["recorded_readings"]}
    assert "frozen_soak_request_count_needed_an_authorized_supplement" in readings
    assert "original_soak_evidence_untouched_by_the_supplement" in readings


def test_soak_was_train_only_and_touched_no_bank(manifest):
    block = manifest["soak"]
    assert block["split"] == "train"
    assert block["setup_source"] == contract.SOURCE_P10D
    assert block["nonbank_train_only"] is True


def test_soak_covered_thousands_of_states_both_colours_and_all_buckets(manifest):
    coverage = manifest["soak"]["coverage"]
    assert coverage["distinct_public_states"] >= 2000
    assert coverage["both_colors_covered"] is True
    assert set(coverage["requests_by_observer_color"]) == {"red", "blue"}
    assert coverage["all_progress_buckets_covered"] is True
    assert set(coverage["requests_by_progress_bucket"]) == {"early", "middle", "late"}


def test_soak_exercised_all_eight_behaviour_strata(manifest):
    coverage = manifest["soak"]["coverage"]
    assert coverage["strata_covered"] == 8
    assert set(coverage["requests_by_stratum"]) == set(contract.OPPONENT_STRATA)


def test_soak_outcomes_are_carried_as_report_only(manifest, acceptance):
    assert "outcomes_report_only" in manifest["soak"]["coverage"]
    readings = {row["reading"] for row in acceptance["recorded_readings"]}
    assert "soak_outcomes_are_report_only" in readings


# ---------------------------------------------------------------------------
# Crash, restart and the store's set algebra
# ---------------------------------------------------------------------------


def test_three_legs_with_three_different_worker_counts(manifest):
    restart = manifest["restart"]
    assert restart["legs"] >= 3
    assert restart["legs_ge_3"] is True
    assert restart["worker_counts_distinct"] is True
    assert len(set(restart["distinct_worker_counts"])) == restart["legs"]


def test_a_real_process_kill_landed_after_committed_work_existed(manifest):
    restart = manifest["restart"]
    assert restart["kill_legs"] >= 1
    assert restart["kill_signal"] == "SIGKILL"
    assert restart["really_signalled"] is True
    assert all(int(value) > 0 for value in restart["committed_before_kill"])
    killed = [leg for leg in manifest["legs"] if "worker_returncode" in leg]
    assert killed and all(leg["worker_returncode"] == -9 for leg in killed)


def test_resume_was_exact_set_subtraction_with_no_recomputation(manifest):
    restart = manifest["restart"]
    assert restart["resume_rule"] == "exact logical request-id set subtraction"
    assert restart["recomputed_on_both_sides"] == 0


def test_the_store_holds_exactly_the_scheduled_ids(manifest):
    reconciliation = manifest["reconciliation"]
    assert reconciliation["missing_request_ids_zero"] is True
    assert reconciliation["duplicate_request_ids_zero"] is True
    assert reconciliation["unscheduled_request_ids_zero"] is True
    assert reconciliation["exactly_scheduled"] is True
    assert reconciliation["scheduled"] == reconciliation["distinct_committed"]


# ---------------------------------------------------------------------------
# The per-request audit
# ---------------------------------------------------------------------------


def test_every_committed_request_was_re_derived(manifest, audit):
    block = audit["per_request_audit"]
    assert block["requests_audited"] == manifest["soak"]["requests"]
    assert block["worlds_verified"] == manifest["soak"]["worlds"]


def test_every_zero_tolerance_counter_is_zero(audit):
    counters = audit["per_request_audit"]["counters"]
    for name in (
        "inventory_errors",
        "public_constraint_errors",
        "provenance_mismatches",
        "hidden_input_accesses",
        "audit_findings",
        "nondeterministic_requests",
    ):
        assert int(counters[name]) == 0, name
    assert audit["per_request_audit"]["distinct_findings"] == []


def test_cross_topology_replay_was_substantial_and_exact(manifest, audit):
    replay = audit["cross_topology_replay"]
    assert replay["exact"] is True
    assert replay["digest_mismatches"] == []
    assert replay["complete"] is True
    assert replay["requests"] >= manifest["soak"]["requests"] // 8
    assert replay["order"] == "reverse"


def test_requests_sharing_one_decision_produced_identical_worlds(audit):
    shared = audit["shared_decision_agreement"]
    assert shared["agree"] is True
    assert shared["disagreements"] == []


def test_the_store_identity_is_content_only(manifest, audit):
    assert manifest["soak"]["store_content_digest"] == audit["store_content_digest"]
    assert len(manifest["soak"]["store_content_digest"]) == 64


# ---------------------------------------------------------------------------
# `phase11_system_v1`
# ---------------------------------------------------------------------------


def test_system_document_fills_exactly_the_template_slots(system):
    document = system["phase11_system_v1"]
    template = _load("agent_01_phase11_contract.json")["documents"]["phase11_system_v1"]
    assert document["system_version"] == template["system_version"]
    assert set(document["filled_slots"]) == {
        slot["slot"] for slot in template["unbound_slots"]
    }


def test_system_document_changes_nothing_that_was_bound(system):
    document = system["phase11_system_v1"]
    template = _load("agent_01_phase11_contract.json")["documents"]["phase11_system_v1"]
    assert document["bound_now"] == template["bound_now"]
    assert document["filling_rules"] == template["filling_rules"]
    assert document["phase12_rule"] == template["phase12_rule"]


def test_bank_slot_resolves_to_the_agent_1_frozen_identities(system):
    filled = system["phase11_system_v1"]["filled_slots"]["bank_digests"]
    for filename, version in (
        ("agent_01_validation_bank.json", "phase11_validation_bank_v1"),
        ("agent_01_test_bank.json", "phase11_test_bank_v1"),
    ):
        bank = _load(filename)
        assert filled[version] == bank["manifest"]["bank_digest"]


def test_system_slots_carry_the_accepted_versions(system):
    filled = system["phase11_system_v1"]["filled_slots"]
    assert filled["evaluator_implementation"]["evaluator_version"] == (
        contract.EVALUATOR_VERSION
    )
    assert filled["sampler_implementation"]["sampler_version"] == "belief_sampler_v1"
    assert filled["information_safety_evidence"]["information_safety_version"] == (
        contract.INFORMATION_SAFETY_VERSION
    )
    assert filled["runtime_benchmark"]["configuration_unchanged"] is True
    assert (
        float(filled["runtime_benchmark"]["measured_p95_forward_64_ms"])
        <= float(filled["runtime_benchmark"]["ceiling_ms"])
        == 500.0
    )


def test_runtime_slot_carries_the_frozen_benchmark_configuration(system):
    configuration = system["phase11_system_v1"]["filled_slots"]["runtime_benchmark"][
        "benchmark_configuration"
    ]
    frozen = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in contract.RUNTIME_BENCHMARK_CONFIGURATION.items()
    }
    assert configuration == frozen


def test_bound_evidence_digests_match_live_artifact_bytes(system):
    filled = system["phase11_system_v1"]["filled_slots"]
    pairs = (
        (
            filled["sampler_implementation"]["audit_evidence_digest"],
            "agent_03_sampler_audit.json",
        ),
        (
            filled["information_safety_evidence"]["attack_evidence_digest"],
            "agent_04_information_safety.json",
        ),
        (
            filled["information_safety_evidence"]["reproducibility_evidence_digest"],
            "agent_04_reproducibility.json",
        ),
        (filled["runtime_benchmark"]["artifact_digest"], "agent_04_runtime.csv"),
    )
    for digest, filename in pairs:
        path = DATA_DIRECTORY / filename
        if not path.exists():
            pytest.skip(f"{filename} is absent")
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest(), filename


def test_system_digest_re_derives_from_the_document(system):
    document = dict(system["phase11_system_v1"])
    recorded = document.pop("system_digest")
    recomputed = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert recomputed == recorded


def test_no_absolute_path_reaches_the_system_identity(system):
    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        elif isinstance(node, str):
            yield node

    for text in walk(system["phase11_system_v1"]):
        assert not text.startswith("/"), text
        assert not text.startswith("~"), text


# ---------------------------------------------------------------------------
# Preservation, sealing and acceptance
# ---------------------------------------------------------------------------


def test_every_preserved_identity_is_exact(system):
    preservation = system["preservation"]
    assert preservation["exact"] is True
    assert preservation["phase9_checkpoint_unchanged"] is True
    assert preservation["belief_head_unchanged"] is True
    assert preservation["phase10_selector_unchanged"] is True
    assert preservation["phase7_library_unchanged"] is True
    assert preservation["optimizer_step_delta"] == 0
    assert preservation["phase11_optimizer_steps"] == 0
    assert preservation["after"]["phase9_parameters"] == 863_959


def test_frozen_inputs_match_the_accepted_contract(manifest):
    frozen = manifest["frozen_inputs"]
    assert frozen["phase9_checkpoint_sha256"] == (
        contract.ACCEPTED_PHASE9_CHECKPOINT_SHA256
    )
    assert frozen["belief_head_digest"] == contract.ACCEPTED_BELIEF_HEAD_DIGEST
    assert frozen["validation_freeze_digest"] == (
        "ad2562af538abc6c78fc5b12bc1f57d3e32184172acde390417a00d500a0d912"
    )


def test_test_bank_scored_access_is_still_zero(acceptance):
    handoff = acceptance["handoff_to_agent_7"]
    assert handoff["test_bank"]["scored_access_so_far"] == 0
    sealing = handoff["test_bank"]["sealing_proof"]
    assert sealing["violations"] == []
    assert sealing["test_bank_structural_only"] is True
    for key in (
        "scored_prediction_total",
        "privileged_truth_total",
        "outcome_total",
        "neural_inference_total",
    ):
        assert int(sealing[key]) == 0, key
    assert sealing["test_refused_without_authorization"] is True


def test_no_forbidden_operation_was_performed(acceptance):
    counters = acceptance["forbidden_operation_counters"]
    assert counters
    assert all(int(value) == 0 for value in counters.values()), counters


def test_the_known_validation_reading_was_carried_not_repaired(acceptance):
    diagnostic = acceptance["diagnostic_carried_forward"]
    assert diagnostic["validation_R_CE"] == pytest.approx(0.9750)
    assert "changes nothing" in diagnostic["reading"]
    counters = acceptance["forbidden_operation_counters"]
    assert int(counters["calibration_operations"]) == 0
    assert int(counters["threshold_changes"]) == 0
    assert int(counters["sampler_rule_changes"]) == 0


def test_the_acceptance_status_agrees_with_its_own_gates(acceptance):
    gates = acceptance["completion_gates"]
    expected = "PASS" if all(gates.values()) else "FAIL"
    assert acceptance["status"] == expected
    assert acceptance["gates_true"] == sum(1 for value in gates.values() if value)
    assert acceptance["gates_total"] == len(gates)
    assert acceptance["false_gates"] == sorted(
        name for name, value in gates.items() if not value
    )


def test_every_gate_other_than_the_self_referential_one_is_true(acceptance):
    """The suite gate is about the run containing this test; the rest are not."""
    false_gates = set(acceptance["false_gates"])
    assert false_gates <= {"full_suite_green"}, sorted(false_gates)
    gates = acceptance["completion_gates"]
    for name in (
        "soak_requests_ge_8192",
        "soak_store_equals_realizable_schedule",
        "every_playable_game_gave_eight_requests",
        "original_requests_preserved_exactly",
        "supplemental_rules_identical_to_frozen",
        "supplemental_playable_games_exact",
        "supplemental_requests_exact",
    ):
        assert gates[name] is True, name


def test_acceptance_carries_the_twenty_three_required_gates(acceptance):
    required = {
        "agents1_5_pass",
        "test_scored_access_zero",
        "soak_requests_ge_8192",
        "soak_nonbank_train_only",
        "thousands_unique_states",
        "both_colors_covered",
        "all_game_progress_buckets_covered",
        "restart_resume_pass",
        "missing_request_ids_zero",
        "duplicate_request_ids_zero",
        "unscheduled_request_ids_zero",
        "inventory_errors_zero",
        "public_constraint_errors_zero",
        "provenance_mismatches_zero",
        "hidden_input_access_zero",
        "deterministic_rebuild_pass",
        "cross_topology_replay_pass",
        "phase11_system_v1_frozen",
        "phase9_checkpoint_unchanged",
        "belief_head_unchanged",
        "phase10_selector_unchanged",
        "no_optimizer_steps",
        "full_suite_green",
    }
    assert len(required) == 23
    assert required <= set(acceptance["completion_gates"])


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    suite = acceptance.get("suite")
    expected = bool(suite) and suite.get("returncode") == 0
    assert acceptance["completion_gates"]["full_suite_green"] == expected


def test_handoff_gives_agent_7_the_system_digest_and_the_gates(acceptance, system):
    handoff = acceptance["handoff_to_agent_7"]
    assert handoff["for_agent"] == 7
    assert handoff["phase11_system_v1_digest"] == (
        system["phase11_system_v1"]["system_digest"]
    )
    assert handoff["final_test_entry_point"] == (
        "stratego.evaluation.phase11_pipeline.run_phase11_pipeline"
    )
    assert len(handoff["hard_gates"]) == 8
    assert handoff["administrative_freeze_requirements"]


# ---------------------------------------------------------------------------
# The materialized-stream collision audit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stream_audit():
    return _load("agent_06_stream_audit.json")


def test_stream_audit_covers_every_materialized_domain(stream_audit):
    assert set(stream_audit["per_domain"]) == {
        "soak_setup",
        "soak_match",
        "world_sample",
        "world_order",
        "world_categorical",
    }
    for name, entry in stream_audit["per_domain"].items():
        assert entry["domain"] == name
        for field in (
            "logical_identities",
            "distinct_derived_seeds",
            "intentional_logical_identity_reuse",
            "internal_accidental_collisions",
            "cross_universe_accidental_collisions",
        ):
            assert field in entry, f"{name}/{field}"


def test_every_domain_is_injective(stream_audit):
    """The condition tested: two different logical identities must not map
    to the same derived 63-bit seed."""
    for name, entry in stream_audit["per_domain"].items():
        assert entry["logical_identities"] == entry["distinct_derived_seeds"], name
        assert entry["internal_accidental_collisions"] == 0, name
        assert entry["cross_universe_accidental_collisions"] == 0, name


def test_no_accidental_collisions_internally_or_across_universes(stream_audit):
    assert stream_audit["total_accidental_collisions"] == 0
    combined = stream_audit["combined"]
    assert combined["no_collisions"] is True
    assert combined["accidental_collisions"] == 0
    assert combined["findings"] == []
    assert combined["unique_logical_identities"] == combined["distinct_seeds"]
    assert combined["bit_width"] == 63
    agent6 = stream_audit["agent6"]
    assert agent6["unique_logical_identities"] == agent6["distinct_seeds"]


def test_the_combined_universe_holds_both_agents(stream_audit):
    agent6 = stream_audit["agent6"]
    combined = stream_audit["combined"]
    assert combined["unique_logical_identities"] == (
        agent6["unique_logical_identities"]
        + stream_audit["agent4"]["universe_identities"]
        - agent6["identities_intentionally_shared_with_agent4"]
    )
    assert agent6["identities_new_relative_to_agent4"] + agent6[
        "identities_intentionally_shared_with_agent4"
    ] == agent6["unique_logical_identities"]


def test_the_agent4_universe_was_reproduced_exactly(stream_audit):
    agent4 = stream_audit["agent4"]
    assert agent4["reproduces_accepted_record"] is True
    recorded = _load("agent_04_stream_audit.json")["collision_audit"]["per_domain"]
    for name, entry in recorded.items():
        assert agent4["per_domain_counts"][name] == entry["identities"], name
    assert agent4["universe_identities"] == sum(
        entry["identities"] for entry in recorded.values()
    )


def test_intentional_reuse_is_deduplicated_not_counted_as_collision(stream_audit):
    reuse = stream_audit["intentional_reuse"]
    assert reuse["deduplicated_before_comparison"] is True
    # Several requests attach to one position; the world identities collapse.
    assert reuse["request_world_identity_pairs"] == reuse["requests"] * 64
    assert reuse["distinct_world_sample_identities"] <= (
        reuse["request_world_identity_pairs"]
    )
    assert reuse["world_identity_pairs_deduplicated"] == (
        reuse["request_world_identity_pairs"]
        - reuse["distinct_world_sample_identities"]
    )
    # The original range's soak identities are Agent 1's, carried by Agent 4.
    assert stream_audit["per_domain"]["soak_match"][
        "intentional_logical_identity_reuse"
    ] == SOAK_GAME_COUNT
    assert stream_audit["per_domain"]["soak_setup"][
        "intentional_logical_identity_reuse"
    ] == SOAK_GAME_COUNT * 2
    # ... and only the supplemental games' identities are new.
    assert stream_audit["per_domain"]["soak_match"][
        "identities_new_relative_to_agent4"
    ] == 29
    assert stream_audit["per_domain"]["soak_setup"][
        "identities_new_relative_to_agent4"
    ] == 58


def test_the_universe_was_reconstructed_from_the_final_schedule(stream_audit):
    fidelity = stream_audit["reconstruction_fidelity"]
    assert fidelity["scheduled_requests"] == SOAK_REQUEST_COUNT == 8192
    assert fidelity["request_count_exact"] is True
    assert fidelity["requests_match_store"] is True
    assert fidelity["original_prefix_requests"] == 7960
    assert fidelity["supplemental_requests"] == 232
    assert fidelity["prefix_and_supplement_sum"] is True
    assert fidelity["original_prefix_represented"] is True
    assert fidelity["supplemental_represented"] is True
    assert fidelity["games"] == SOAK_GAME_COUNT + 29
    assert fidelity["supplemental_games"] == 29
    assert fidelity["reconstructed_identities_match_schedule"] is True


def test_every_materialized_world_token_is_represented(stream_audit):
    fidelity = stream_audit["reconstruction_fidelity"]
    assert fidelity["every_used_token_represented"] is True
    assert fidelity["world_tokens_used_by_requests"] == (
        fidelity["world_tokens_enumerated"]
    )
    assert stream_audit["per_domain"]["world_sample"]["logical_identities"] == (
        fidelity["world_tokens_enumerated"]
    )


def test_the_bulk_derivation_path_agrees_with_the_public_helpers(stream_audit):
    check = stream_audit["reconstruction_fidelity"]["fast_path_check"]
    assert check["exact"] is True
    assert check["mismatches"] == 0
    assert check["derivations_checked"] > 0


def test_the_two_reconciliation_gates_are_true(acceptance):
    gates = acceptance["completion_gates"]
    assert gates["agent6_materialized_stream_collisions_zero"] is True
    assert gates["agent6_stream_universe_reconstruction_faithful"] is True
    assert int(
        acceptance["forbidden_operation_counters"]["accidental_stream_seed_collisions"]
    ) == 0
