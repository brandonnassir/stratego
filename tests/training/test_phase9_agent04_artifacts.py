"""Phase 9 Agent 4: the published artifacts say what the audit actually did.

These tests read the reports rather than recompute them. Their job is to stop a
published artifact from drifting away from the frozen contract it claims to
satisfy — an acceptance file recording a gate it never measured, an exhaustive
audit that quietly stopped being exhaustive, an anti-leak run whose "zero
mismatches" came from auditing nothing, or a positive control counted as fired
without firing.

The artifacts are only present after `scripts/run_phase9_agent04.py` has run,
so every test skips cleanly when they are absent rather than failing a fresh
checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratego.training import phase9_antileak as antileak
from stratego.training import phase9_contract as contract
from stratego.training import phase9_targets as targets

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "phase_9_data"
ACCEPTANCE = DATA_DIRECTORY / "agent_04_acceptance.json"
TARGET_AUDIT = DATA_DIRECTORY / "agent_04_target_audit.json"
ANTILEAK = DATA_DIRECTORY / "agent_04_antileak.json"
EXAMPLE_CONTRACT = DATA_DIRECTORY / "agent_04_example_contract.json"

ANCHOR_SHA256 = "f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca"
ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)

#: Assignment floors. A run below either is not the audit that was asked for.
ANTILEAK_TRIAL_MINIMUM = 25_000
BEHAVIOR_RECHECK_MINIMUM = 100_000

#: `full_suite_green` is deliberately excluded from the gate assertions below.
#: A test that runs *inside* the suite cannot soundly assert that the suite
#: passed: the assertion is evaluated before its own run finishes. That gate is
#: established by `--record-final-suite`, which re-runs the suite with the
#: artifacts present.
SELF_REFERENTIAL_GATE = "full_suite_green"


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture
def acceptance():
    return _load(ACCEPTANCE)


@pytest.fixture
def audit():
    return _load(TARGET_AUDIT)


@pytest.fixture
def leak():
    return _load(ANTILEAK)


@pytest.fixture
def example_contract():
    return _load(EXAMPLE_CONTRACT)


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def test_acceptance_reports_every_measured_gate_true(acceptance):
    failed = [
        name
        for name, value in acceptance["completion_gates"].items()
        if not value and name != SELF_REFERENTIAL_GATE
    ]
    assert not failed, f"gates reported false: {failed}"
    assert not acceptance["problems"]
    assert acceptance["gates_total"] == len(acceptance["completion_gates"])


def test_every_assigned_completion_gate_is_present(acceptance):
    """The gate list the assignment requires, checked name by name."""
    required = {
        "agents1_3_pass",
        "corpus_resolver_verified",
        "corpus_digests_match",
        "same_player_sequence_audit_pass",
        "red_blue_perspective_audit_pass",
        "advantages_exhaustively_match",
        "wdl_targets_exhaustively_match",
        "advantage_filter_exact",
        "value_target_simplex_failures_zero",
        "belief_target_mismatches_zero",
        "behavior_reproduction_ge_100k",
        "behavior_reproduction_mismatches_zero",
        "hidden_permutation_trials_ge_25000",
        "model_input_leak_mismatches_zero",
        "positive_controls_fire",
        "learner_control_mismatches_zero",
        "no_meaningful_rl_training",
        "full_suite_green",
    }
    assert required <= set(acceptance["completion_gates"])


def test_the_recorded_suite_ran_with_the_artifacts_in_place(acceptance):
    """The recorded result must come from a pass that could see the artifacts.

    The harness runs before the artifacts exist, so in that pass every test in
    this file skips and a recorded green would mean nothing. Only the
    `--record-final-suite` pass sets this flag.
    """
    assert acceptance["tests_after"].get("covers_agent_04_artifact_tests"), (
        "the recorded suite ran before the artifacts existed, so it never "
        "exercised the artifact tests"
    )


def test_the_prerequisite_chain_is_recorded_as_accepted(acceptance):
    prerequisites = acceptance["prerequisites"]
    assert prerequisites["agent1_status"] == "PASS"
    assert prerequisites["agent2_status"] == "PASS"
    assert prerequisites["agent3_status"] == "PASS"
    assert prerequisites["contract_digest"] == ACCEPTED_CONTRACT_DIGEST
    assert prerequisites["phase8_checkpoint_sha256"] == ANCHOR_SHA256
    assert not prerequisites["problems"]


def test_the_audited_rollout_is_the_one_agent_3_sealed(acceptance):
    rollout = acceptance["prerequisites"]["audited_rollout"]
    assert rollout["state"] != "COLLECTING"
    assert rollout["digest_matches_agent_3"], (
        "the audited bytes are not the sealed rollout Agent 3 recorded"
    )
    assert rollout["behavior_checkpoint_sha256"] == ANCHOR_SHA256
    assert rollout["games"] > 0


def test_the_corpus_resolver_was_verified_without_a_hard_coded_path(acceptance):
    corpus = acceptance["corpus"]
    assert corpus["resolver"].endswith("default_corpus_root()")
    assert corpus["resolved_root_matches_expected"]
    assert corpus["identity_matches"]
    assert corpus["observed_identity"] == corpus["accepted_identity"]
    assert not corpus["modules_hard_coding_absolute_paths"]


# ---------------------------------------------------------------------------
# The exhaustive target audit
# ---------------------------------------------------------------------------


def test_the_audit_covered_every_learner_decision_of_the_rollout(audit):
    assert audit["games_audited"] >= 2048
    assert audit["learner_decisions"] > 0
    assert audit["examples_audited"] == audit["learner_decisions"], (
        "an audited rollout must build exactly one example per learner decision"
    )


def test_the_audit_reports_no_mismatch_of_any_kind(audit):
    for key in (
        "advantage_mismatches",
        "wdl_target_mismatches",
        "value_target_simplex_failures",
        "belief_target_mismatches",
        "eligibility_mismatches",
        "standardization_mismatches",
        "sequence_problem_count",
        "behavior_quantity_problem_count",
        "example_problem_count",
    ):
        assert audit[key] == 0, f"{key} = {audit[key]}"
    assert not audit["filter_problems"]


def test_all_three_learner_control_modes_and_both_colours_were_audited(audit):
    assert sorted(audit["learner_control_counts"]) == ["blue", "both", "red"]
    assert audit["learner_decisions_by_colour"]["red"] > 0
    assert audit["learner_decisions_by_colour"]["blue"] > 0
    assert sorted(audit["bucket_counts"]) == ["current", "historical", "rule", "stress"]


def test_the_recorded_filter_matches_the_frozen_contract(audit):
    statistics = audit["statistics"]
    assert statistics["quantile"] == contract.ADVANTAGE_FILTER_QUANTILE
    assert statistics["floor"] == contract.ADVANTAGE_FILTER_FLOOR
    assert statistics["standardization_epsilon"] == contract.ADVANTAGE_STANDARDIZATION_EPSILON
    assert statistics["threshold"] >= contract.ADVANTAGE_FILTER_FLOOR
    assert statistics["advantage_version"] == contract.PHASE9_ADVANTAGE_VERSION


def test_the_independent_reference_agrees_with_the_production_filter(audit):
    statistics = audit["statistics"]
    reference = audit["reference"]
    assert statistics["threshold"] == pytest.approx(reference["threshold"], abs=1e-12)
    assert statistics["eligible"] == reference["eligible"]
    assert statistics["mean_eligible"] == pytest.approx(reference["mean_eligible"], abs=1e-12)
    assert statistics["std_eligible"] == pytest.approx(reference["std_eligible"], abs=1e-12)


def test_the_retention_fraction_is_the_quantile_it_claims(audit):
    """A Q75 filter over a large sample retains a quarter of it, near exactly."""
    assert audit["retention_fraction"] == pytest.approx(0.25, abs=0.005)
    assert audit["eligible_examples"] == audit["statistics"]["eligible"]


def test_belief_supervision_is_actually_present(audit):
    """Zero supervised squares would make "no belief mismatch" vacuous."""
    assert audit["supervised_belief_squares"] > 0
    assert audit["mean_supervised_belief_squares"] > 1.0


# ---------------------------------------------------------------------------
# Anti-leak
# ---------------------------------------------------------------------------


def test_the_anti_leak_run_met_its_floor(leak):
    assert leak["valid_trials"] >= ANTILEAK_TRIAL_MINIMUM
    assert leak["assignment_changed_trials"] > 0, (
        "no trial actually reassigned an identity, so nothing was proven"
    )
    assert leak["mean_hidden_pieces"] > 1.0


def test_no_public_field_moved_in_any_trial(leak):
    assert leak["invariant_mismatches"] == 0
    assert leak["label_control_failures"] == 0
    assert not leak["mismatch_examples"]
    assert leak["model_input_boundary_problem_count"] == 0
    assert not leak["object_graph_problems"]


def test_the_invariant_surface_covers_the_assignment(leak):
    """Observation, legality, frame, learner side, PPO inputs, belief mask."""
    invariant = set(leak["invariant_fields"])
    assert {
        "observation",
        "legal_mask",
        "sampled_action_model",
        "behavior_action_probability",
        "behavior_legal_probabilities",
        "advantage",
        "standardized_advantage",
        "ppo_eligible",
        "wdl_target",
        "belief_mask",
        "learner_side",
    } <= invariant
    assert "belief_target" not in invariant
    assert leak["privileged_fields"] == ["belief_target"]


def test_every_positive_control_fired(leak):
    assert leak["all_positive_controls_fire"]
    assert leak["positive_controls_fired"] == len(antileak.POSITIVE_CONTROL_NAMES)
    fired = {control["control"] for control in leak["positive_controls"] if control["fired"]}
    assert fired == set(antileak.POSITIVE_CONTROL_NAMES)
    for control in leak["positive_controls"]:
        assert control["problems"], f"{control['control']} fired without a finding"


# ---------------------------------------------------------------------------
# Behavior consistency
# ---------------------------------------------------------------------------


def test_the_behavior_recheck_met_its_floor_and_found_nothing(acceptance):
    behavior = acceptance["behavior_consistency"]
    assert behavior["learner_decisions_rechecked"] >= BEHAVIOR_RECHECK_MINIMUM
    assert behavior["learner_mismatches"] == 0
    assert behavior["legal_set_mismatches"] == 0
    assert behavior["action_redraw_mismatches"] == 0
    assert behavior["policy_token_mismatches"] == 0
    assert (
        behavior["max_abs_probability_difference"]
        <= contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE
    )


def test_the_recheck_was_run_against_the_frozen_anchor_and_did_not_move_it(acceptance):
    behavior = acceptance["behavior_consistency"]
    assert behavior["behavior_checkpoint_sha256"] == ANCHOR_SHA256
    assert behavior["snapshot_weights_unchanged"]


def test_the_wrong_checkpoint_control_still_has_teeth(acceptance):
    """A re-check that cannot fail proves nothing about the one that passed."""
    control = acceptance["behavior_consistency"]["untrained_checkpoint_control"]
    assert control["control_holds"]
    assert control["decisions"] > 0
    assert (
        control["max_abs_probability_difference"]
        > contract.BEHAVIOR_PROBABILITY_ABS_TOLERANCE
    )


# ---------------------------------------------------------------------------
# The example contract and the train order
# ---------------------------------------------------------------------------


def test_the_published_contract_is_the_live_one(example_contract, acceptance):
    assert example_contract["example_contract_digest"] == targets.example_contract_digest()
    assert acceptance["example_contract_digest"] == targets.example_contract_digest()
    assert example_contract["example_contract"] == targets.example_contract()
    assert acceptance["example_version"] == targets.PHASE9_EXAMPLE_VERSION


def test_the_published_contract_names_one_model_input(example_contract):
    document = example_contract["example_contract"]
    assert document["model_input_fields"] == ["observation"]
    assert document["fields"]["belief_target"]["role"] == "loss_input"
    assert document["populations"]["value"] == "every learner decision"


def test_the_train_order_demonstration_actually_ran(example_contract):
    order = example_contract["train_order"]
    assert order["universe"] > 0
    assert order["minibatch_size"] == contract.MINIBATCH_SIZE
    assert order["epochs_per_rollout"] == contract.EPOCHS_PER_ROLLOUT
    assert order["epoch_orders_differ"]
    assert order["epoch_order_reproducible"]
    assert order["resume_reproduces_interrupted_order"]
    assert order["epoch0_seed"] != order["epoch1_seed"]
    assert not example_contract["problems"]


def test_the_final_partial_minibatch_is_not_dropped(example_contract):
    order = example_contract["train_order"]
    covered = (order["minibatches_per_epoch"] - 1) * order["minibatch_size"] + (
        order["final_minibatch_size"]
    )
    assert covered == order["universe"]


# ---------------------------------------------------------------------------
# Mission boundaries
# ---------------------------------------------------------------------------


def test_nothing_in_the_target_path_can_train(acceptance):
    training = acceptance["no_training_audit"]
    assert training["no_meaningful_rl_training"]
    assert not training["symbol_findings"]
    assert training["trainable_parameters"] == 0
    assert training["weights_unchanged"]


def test_the_handoff_to_agent_5_names_what_agent_5_needs(acceptance):
    handoff = acceptance["handoff_to_agent_5"]
    for key in (
        "example_iterator",
        "train_order",
        "cursor",
        "ppo_eligibility",
        "standardized_advantages",
        "behavior_quantity",
        "wdl_targets",
        "belief_targets",
        "model_input_boundary",
    ):
        assert handoff.get(key), f"handoff is missing {key}"
