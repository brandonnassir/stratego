"""Phase 9 Agent 5: the published artifacts say what the run actually did.

These tests read the reports rather than recompute them. Their job is to stop a
published artifact from drifting away from the frozen contract it claims to
satisfy — a soak that reports "no non-finite losses" because it ran no updates,
a resume validation whose MPS envelope was never measured, a "swapped bindings
fail" claim from a fixture whose two checkpoints were the same file, or a
selection statement on a run that quietly did select something.

The artifacts exist only after `scripts/run_phase9_agent05.py` has run, so
every test skips cleanly when they are absent rather than failing a fresh
checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratego.training import phase9_checkpoint as pck
from stratego.training import phase9_trainer as pt
from stratego.training.phase9_contract import (
    CHECKPOINT_REQUIRED_FIELDS,
    CLIP_FRACTION_HARD_LIMIT,
    EPOCHS_PER_ROLLOUT,
    KL_HARD_LIMIT,
    MINIBATCH_SIZE,
    PILOT_CANDIDATES,
    contract_digest,
)
from stratego.training.phase9_targets import example_contract_digest

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "phase_9_data"
ACCEPTANCE = DATA_DIRECTORY / "agent_05_acceptance.json"
TRAINER_CONTRACT = DATA_DIRECTORY / "agent_05_trainer_contract.json"
RESUME = DATA_DIRECTORY / "agent_05_resume_validation.json"
SOAK = DATA_DIRECTORY / "agent_05_stability_soak.json"
BENCHMARK = DATA_DIRECTORY / "agent_05_training_benchmark.csv"

ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_EXAMPLE_DIGEST = (
    "a6b17a94449ab764d4b5dd054d677096adfa70c52631865499a60a7a3f44af61"
)

#: Assignment floors. A run below either is not the soak that was asked for.
SOAK_MINIMUM_UPDATES = 2000
SOAK_MINIMUM_ITERATIONS = 3

#: A test that runs *inside* the suite cannot soundly assert that the suite
#: passed. That gate is established by `--record-final-suite`, which re-runs
#: the suite with the artifacts present.
SELF_REFERENTIAL_GATE = "full_suite_green"


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture
def acceptance():
    return _load(ACCEPTANCE)


@pytest.fixture
def trainer_contract():
    return _load(TRAINER_CONTRACT)


@pytest.fixture
def resume():
    return _load(RESUME)


@pytest.fixture
def soak():
    return _load(SOAK)


# ---------------------------------------------------------------------------
# The contract artifact
# ---------------------------------------------------------------------------


def test_contract_pins_the_accepted_upstream_digests(trainer_contract):
    prerequisites = trainer_contract["prerequisites"]
    assert prerequisites["contract_digest"] == ACCEPTED_CONTRACT_DIGEST
    assert prerequisites["example_contract_digest"] == ACCEPTED_EXAMPLE_DIGEST
    assert prerequisites["phase8_checkpoint_matches_accepted"]
    assert all(
        entry["status"] == "PASS" for entry in prerequisites["acceptances"].values()
    )


def test_contract_digests_still_match_this_build(trainer_contract):
    """A later edit to a frozen module would break this, which is the point."""
    assert contract_digest() == trainer_contract["prerequisites"]["contract_digest"]
    assert (
        example_contract_digest()
        == trainer_contract["prerequisites"]["example_contract_digest"]
    )


def test_contract_records_the_corpus_by_identity_not_location(trainer_contract):
    corpus = trainer_contract["corpus"]
    assert corpus["identity_matches"]
    assert corpus["modules_hard_coding_absolute_paths"] == []
    assert corpus["resolver"].endswith("default_corpus_root()")


def test_contract_states_the_frozen_optimizer_constants(trainer_contract):
    constraints = trainer_contract["trainer"]["optimizer_constraints"]
    assert constraints["minibatch_size"] == MINIBATCH_SIZE
    assert constraints["epochs_per_rollout"] == EPOCHS_PER_ROLLOUT
    assert constraints["gradient_clip_norm"] == 1.0
    assert constraints["weight_decay"] == 0.01
    assert constraints["precision"] == "float32"
    assert trainer_contract["loss"]["ppo"]["clip_epsilon"] == 0.20
    assert trainer_contract["loss"]["value"]["weight"] == 0.5
    assert trainer_contract["loss"]["belief"]["weight"] == 0.25


def test_contract_publishes_all_six_candidates_and_no_seventh(trainer_contract):
    candidates = trainer_contract["pilot_candidate_constructor"]["candidates"]
    assert len(candidates) == len(PILOT_CANDIDATES) == 6
    published = {
        (entry["candidate_id"], entry["learning_rate"], entry["initial_kl_beta"])
        for entry in candidates
    }
    frozen = {
        (entry["candidate_id"], entry["learning_rate"], entry["initial_kl_beta"])
        for entry in PILOT_CANDIDATES
    }
    assert published == frozen
    assert len({entry["train_config_digest"] for entry in candidates}) == 6


def test_the_soak_configuration_is_not_a_pilot_candidate_run(trainer_contract):
    """The soak's identity must not be mistakable for a P9-C pilot run."""
    soak_digest = trainer_contract["soak_configuration"]["train_config_digest"]
    candidates = trainer_contract["pilot_candidate_constructor"]["candidates"]
    assert soak_digest not in {entry["train_config_digest"] for entry in candidates}
    assert trainer_contract["soak_configuration"]["scope"] == pt.SCOPE_SOAK


def test_contract_checkpoint_fields_match_the_frozen_requirement(trainer_contract):
    assert trainer_contract["checkpoint"]["required_fields"] == list(
        CHECKPOINT_REQUIRED_FIELDS
    )
    assert set(pck.PHASE9_CHECKPOINT_KEYS) >= set(CHECKPOINT_REQUIRED_FIELDS)


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------


def test_cpu_resume_is_bitwise_exact_across_processes(resume):
    cpu = resume["devices"]["cpu"]
    assert cpu["batch_identities_equal_every_step"]
    assert cpu["exact_next_batch_after_resume"]
    assert cpu["logical_state_summaries_equal"]
    assert cpu["parameters_end_vs_donor"]["all_exactly_equal"]
    assert cpu["parameters_end_vs_donor"]["max_abs_diff"] == 0.0
    assert cpu["passed"]


def test_cpu_resume_actually_ran_a_split(resume):
    """A "bitwise identical" claim from zero compared steps proves nothing."""
    cpu = resume["devices"]["cpu"]
    assert cpu["compared_steps"] == cpu["total_updates"]
    assert cpu["compared_steps"] > cpu["split_at"] > 0
    assert cpu["parameters_end_vs_donor"]["tensors_compared"] > 0


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_every_logical_quantity_survives_the_reload(resume, device):
    evidence = resume["devices"].get(device)
    if evidence is None:
        pytest.skip(f"the {device} resume leg was not run")
    for field, equal in evidence["logical_state_fields"].items():
        assert equal, field
    assert evidence["learning_rates_equal_every_step"]
    assert evidence["cursor_positions_equal_every_step"]
    assert evidence["global_steps_equal_every_step"]


def test_mps_resume_uses_the_backend_aware_criterion(resume):
    mps = resume["devices"].get("mps")
    if mps is None:
        pytest.skip("the MPS resume leg was not run")
    assert mps["batch_identities_equal_every_step"]
    assert mps["exact_next_batch_after_resume"]
    assert mps["logical_state_summaries_equal"]
    # The resume boundary itself, against a donor that entered the step from
    # bit-identical state, must meet the original tolerances.
    assert mps["parameters_first_post_resume_step"]["all_allclose_rtol1e-5_atol1e-6"]
    assert mps["passed"]


def test_mps_envelope_is_measured_not_assumed(resume):
    """The control legs are what make the backend-aware criterion honest."""
    mps = resume["devices"].get("mps")
    if mps is None:
        pytest.skip("the MPS resume leg was not run")
    control = mps["backend_control"]["end"]
    assert control["tensors_compared"] > 0
    ratio = mps["envelope_ratio_end_vs_donor_over_control"]
    assert ratio is not None
    assert ratio <= resume["criterion"]["tolerances"]["mps_envelope_ratio_limit"]


def test_resume_read_a_real_sealed_rollout(resume):
    rollout = resume["rollout"]
    assert rollout["learner_decisions"] > 0
    assert rollout["games"] > 0
    assert len(rollout["sealed_rollout_digest"]) == 64
    assert rollout["verifications"]["digest_recomputed_from_commits"]
    assert rollout["verifications"]["learner_control_mismatches"] == 0


# ---------------------------------------------------------------------------
# The soak
# ---------------------------------------------------------------------------


def test_soak_met_the_assignment_floors(soak):
    assert soak["totals"]["optimizer_updates"] >= SOAK_MINIMUM_UPDATES
    assert soak["totals"]["rl_iterations"] >= SOAK_MINIMUM_ITERATIONS
    assert soak["meets_minimum_updates"]


def test_soak_zero_counters_come_from_a_real_run(soak):
    """Zero failures are only meaningful with a denominator."""
    assert soak["totals"]["examples_consumed"] > 0
    assert soak["totals"]["learner_decisions"] > 0
    assert soak["non_finite_metric_rows"] == 0
    for name, value in soak["required_zero_counters"].items():
        assert value == 0, name


def test_soak_stayed_inside_the_frozen_instability_limits(soak):
    stability = soak["stability"]
    assert stability["kl_hard_limit"] == KL_HARD_LIMIT
    assert stability["clip_fraction_hard_limit"] == CLIP_FRACTION_HARD_LIMIT
    assert not stability["kl_hard_limit_exceeded"]
    assert not stability["clip_fraction_hard_limit_exceeded"]
    assert stability["max_epoch_mean_kl"] <= KL_HARD_LIMIT
    assert stability["max_epoch_clip_fraction"] <= CLIP_FRACTION_HARD_LIMIT


def test_soak_recorded_every_required_diagnostic(soak):
    stability = soak["stability"]
    for field in (
        "mean_behavior_kl",
        "mean_clip_fraction",
        "mean_policy_entropy",
        "mean_advantage_retention",
        "mean_grad_norm_pre_clip",
        "kl_beta_final",
    ):
        assert field in stability, field
    totals = soak["totals"]
    for field in (
        "examples_per_second",
        "updates_per_second",
        "data_wait_seconds",
        "peak_rss_mib",
        "peak_mps_mib",
    ):
        assert field in totals, field


def test_every_soak_iteration_trained_a_distinct_sealed_rollout(soak):
    """One iteration, one behavior snapshot, one sealed rollout — never reused."""
    iterations = soak["iterations"]
    digests = [entry["sealed_rollout_digest"] for entry in iterations]
    snapshots = [entry["behavior_snapshot_id"] for entry in iterations]
    assert len(set(digests)) == len(digests)
    assert len(set(snapshots)) == len(snapshots)
    assert snapshots == [f"B{entry['iteration']:03d}" for entry in iterations]
    for entry in iterations:
        assert entry["updates"] > 0
        assert entry["learner_decisions"] > 0


def test_no_iteration_trained_on_a_stale_behavior_checkpoint(soak):
    """Each iteration's rollout was collected by the weights that trained on it.

    Iteration 1 starts from the Phase 8 anchor and every later iteration's
    behavior checkpoint must be a *new* digest, which is what distinguishes an
    on-policy sequence from replaying one rollout five times.
    """
    digests = [entry["behavior_checkpoint_sha256"] for entry in soak["iterations"]]
    assert len(set(digests)) == len(digests)


def test_the_soak_selected_nothing(soak):
    statement = soak["selection_statement"]
    assert statement["scope"] == pt.SCOPE_SOAK
    assert statement["selects_a_configuration"] is False
    assert statement["validation_bank_opened"] is False
    assert statement["final_test_bank_opened"] is False
    assert statement["validation_score_computed"] is False
    assert statement["weights_carried_into_agent_6"] is False


def test_the_soak_ran_outside_every_production_rollout_namespace(soak):
    """Agent 6 and 7 must start from namespaces nothing here has written."""
    assert "agent_05_soak" in soak["selection_statement"]["rollout_root"]


def test_the_kl_controller_history_is_one_entry_per_epoch(soak):
    history = soak["stability"]["kl_controller_history"]
    assert len(history) == soak["totals"]["rl_iterations"] * EPOCHS_PER_ROLLOUT
    for entry in history:
        assert entry["direction"] in ("increase", "decrease", "unchanged")
        assert 1e-4 <= entry["beta_after"] <= 0.2


# ---------------------------------------------------------------------------
# The archive and the two-checkpoint binding fixture
# ---------------------------------------------------------------------------


def test_the_soak_produced_a_real_namespace_local_archive_member(soak):
    members = soak["archive_members"]
    assert members, "the frozen cadence should have archived at least one member"
    for member in members:
        assert member["qualified_identity"] == (
            f"{member['namespace']}|{member['local_identity']}"
        )
        assert member["namespace"] in member["policy_token"]
        assert len(member["checkpoint_sha256"]) == 64
        assert member["rl_iteration"] % 5 == 0


def test_the_archive_member_answers_a_real_iteration_6_schedule(soak):
    """Agent 3's carry-forward, closed with real weights rather than an assertion."""
    rehearsal = soak.get("archive_rehearsal")
    if rehearsal is None or rehearsal.get("skipped"):
        pytest.skip("the archive rehearsal was not run")
    assert "H005" in rehearsal["active_window"]
    assert rehearsal["games_against_H005"] > 0
    assert rehearsal["opponent_digest_is_the_archived_member"]
    assert rehearsal["sealed"] is False


def test_the_binding_fixture_used_two_genuinely_different_checkpoints(soak):
    fixture = soak.get("binding_fixture")
    if fixture is None:
        pytest.skip("the binding fixture has not been produced yet")
    assert fixture["checkpoints_are_genuinely_different"]
    assert (
        fixture["learner"]["checkpoint_sha256"]
        != fixture["historical_opponent"]["checkpoint_sha256"]
    )
    assert (
        fixture["learner"]["state_dict_digest"]
        != fixture["historical_opponent"]["state_dict_digest"]
    )


def test_each_side_verified_against_its_own_checkpoint(soak):
    fixture = soak.get("binding_fixture")
    if fixture is None:
        pytest.skip("the binding fixture has not been produced yet")
    own = fixture["each_side_against_its_own_checkpoint"]
    assert own["all_verified"]
    assert own["learner"]["decisions"] > 0
    assert own["historical_opponent"]["decisions"] > 0
    assert own["learner"]["failed"] == 0
    assert own["historical_opponent"]["failed"] == 0


def test_swapping_the_checkpoint_bindings_failed_the_verification(soak):
    fixture = soak.get("binding_fixture")
    if fixture is None:
        pytest.skip("the binding fixture has not been produced yet")
    swapped = fixture["swapped_bindings"]
    assert swapped["all_failed"]
    for side in (
        "learner_decisions_against_opponent_checkpoint",
        "opponent_decisions_against_learner_checkpoint",
    ):
        assert swapped[side]["decisions"] > 0
        assert swapped[side]["verified"] == 0
        # A swap must fail numerically, not merely by a digest string.
        assert swapped[side]["max_abs_difference"] > 1e-3
    assert fixture["digest_guard_alone"]["rejected_before_any_forward_pass"]


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def test_throughput_proves_topology_cannot_change_a_minibatch(soak):
    throughput = soak.get("throughput")
    if throughput is None:
        pytest.skip("the throughput probe has not been produced yet")
    assert throughput["identical_logical_minibatch_identities"]
    assert throughput["identical_across_synchronization"]
    assert len({entry["workers"] for entry in throughput["measurements"]}) >= 2
    # Losses are deliberately *not* asserted equal here. The claim the frozen
    # train order makes is about which examples a minibatch holds and in what
    # order, and that is what the batch digest measures. On MPS two runs over
    # bit-identical inputs still differ in the last float bits, so requiring
    # equal losses would be asserting backend determinism, which Phase 8
    # already measured to be false on this stack.
    assert "identical_losses_across_topologies" in throughput


def test_throughput_reports_a_rate_that_is_not_synchronization_inflated(soak):
    """The per-phase split costs throughput; both numbers are published."""
    throughput = soak.get("throughput")
    if throughput is None:
        pytest.skip("the throughput probe has not been produced yet")
    for entry in throughput["measurements"]:
        assert entry["examples_per_second"] > 0
        assert entry["unsynchronized_examples_per_second"] > 0


def test_the_benchmark_csv_splits_a_complete_iteration(soak):
    if not BENCHMARK.exists():
        pytest.skip("agent_05_training_benchmark.csv has not been produced yet")
    import csv

    with BENCHMARK.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    iteration_rows = [row for row in rows if row["measurement"] == "soak_iteration"]
    assert iteration_rows
    for row in iteration_rows:
        for column in (
            "collection_seconds",
            "target_construction_seconds",
            "data_wait_seconds",
            "forward_seconds",
            "backward_seconds",
            "checkpoint_seconds",
            "train_seconds",
        ):
            assert row[column] != "", column
        assert float(row["train_seconds"]) > 0


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def test_acceptance_reports_every_completion_gate(acceptance):
    required = {
        "agents1_4_pass",
        "corpus_resolver_verified",
        "corpus_digests_match",
        "ppo_loss_matches_contract",
        "illegal_logit_masking_pass",
        "value_loss_matches_contract",
        "belief_loss_matches_contract",
        "kl_direction_and_beta_controller_pass",
        "entropy_schedule_pass",
        "opponent_only_gradients_zero",
        "cpu_resume_pass",
        "mps_backend_aware_resume_pass",
        "atomic_checkpoint_tests_pass",
        "soak_updates_ge_2000",
        "nonfinite_zero",
        "illegal_targets_zero",
        "identity_mismatches_zero",
        "kl_hard_limit_not_exceeded",
        "clip_fraction_hard_limit_not_exceeded",
        "throughput_measured",
        "no_pilot_selection",
        "no_final_test_access",
        "full_suite_green",
    }
    assert required <= set(acceptance["gates"])


def test_every_gate_except_the_self_referential_one_passed(acceptance):
    failed = [
        name
        for name, value in acceptance["gates"].items()
        if not value and name != SELF_REFERENTIAL_GATE
    ]
    assert failed == []


def test_acceptance_records_the_frozen_versions(acceptance):
    assert acceptance["trainer_version"] == "phase9_trainer_v1"
    assert acceptance["checkpoint_version"] == "phase9_checkpoint_v1"
    assert acceptance["prerequisites"]["contract_digest"] == ACCEPTED_CONTRACT_DIGEST


def test_acceptance_does_not_claim_a_pilot_result(acceptance):
    """Agent 5 hands Agent 6 machinery, never a winner."""
    text = json.dumps(acceptance).lower()
    for forbidden in ("winner", "selected_candidate", "best_candidate"):
        assert forbidden not in text
    assert acceptance["selection_statement"]["selects_a_configuration"] is False
