"""Phase 9: `phase9_operational_amendment_v2` stays inside its reviewed scope.

A second amendment to the same number is exactly where an operational budget
starts to look like a negotiable one. These tests hold the line the review
drew: the ceiling moved and nothing else did; both earlier identities — the
frozen contract's 12 hours and v1's 15 hours — are still readable and still
hash to what they always did; and the scientific quantities are read from the
live contract rather than restated here, so a constant that drifts cannot
hide behind this amendment's paperwork.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from stratego.training import phase9_amendment as v1
from stratego.training import phase9_amendment_v2 as v2
from stratego.training.phase9_contract import (
    ARCHIVE_CADENCE_ITERATIONS,
    CANONICAL_GAMES_PER_ITERATION,
    CANONICAL_ITERATIONS,
    CANONICAL_WALL_CLOCK_CEILING_HOURS,
    EPOCHS_PER_ROLLOUT,
    VALIDATION_CADENCE_ITERATIONS,
    contract_digest,
)

ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)
ACCEPTED_V1_DIGEST = (
    "ee4b05078c676128f78c8e5c31bd10ce4f0841e34a57c4c7c3fca6616e083ac4"
)


# ---------------------------------------------------------------------------
# Nothing underneath the amendment moved
# ---------------------------------------------------------------------------


def test_both_earlier_identities_are_preserved_unedited():
    assert contract_digest() == ACCEPTED_CONTRACT_DIGEST
    assert v1.amendment_digest() == ACCEPTED_V1_DIGEST
    assert CANONICAL_WALL_CLOCK_CEILING_HOURS == 12
    assert v1.AMENDED_CEILING_HOURS == 15
    assert v1.AMENDED_CEILING_SECONDS == 54_000


def test_verify_chain_untouched_reports_no_problems():
    assert v2.verify_chain_untouched() == []


def test_the_chain_names_what_it_amends():
    document = v2.amendment_document()
    assert document["amends"]["amendment_version"] == "phase9_operational_amendment_v1"
    assert document["amends"]["amendment_digest"] == ACCEPTED_V1_DIGEST
    assert document["amends"]["base_contract_digest"] == ACCEPTED_CONTRACT_DIGEST
    assert document["amends"]["in_place_edit"] is False


def test_ceiling_history_keeps_every_value_this_phase_has_held():
    history = v2.ceiling_history()
    assert [entry["seconds"] for entry in history] == [43_200, 54_000, 86_400]
    assert [entry["hours"] for entry in history] == [12, 15, 24]
    assert history[0]["authority"] == "phase9_rl_contract_v1"
    assert history[1]["authority"] == "phase9_operational_amendment_v1"
    assert history[2]["authority"] == "phase9_operational_amendment_v2"
    assert history[0]["digest"] == ACCEPTED_CONTRACT_DIGEST
    assert history[1]["digest"] == ACCEPTED_V1_DIGEST


# ---------------------------------------------------------------------------
# The one thing it changes
# ---------------------------------------------------------------------------


def test_amendment_declares_the_reviewed_ceiling():
    assert v2.AMENDED_CEILING_HOURS == 24
    assert v2.AMENDED_CEILING_SECONDS == 86_400
    assert v2.amended_ceiling_seconds() == 86_400
    change = v2.amendment_document()["change"]
    assert change["field"] == "wall_clock_ceiling_hours"
    assert (change["from_seconds"], change["to_seconds"]) == (54_000, 86_400)
    assert change["historical_values_preserved"] is True


def test_amendment_digest_hashes_its_own_document():
    canonical = json.dumps(
        v2.amendment_document(), sort_keys=True, separators=(",", ":")
    )
    assert v2.amendment_digest() == hashlib.sha256(canonical.encode()).hexdigest()


def test_apply_changes_only_the_ceiling():
    document = {"wall_clock_ceiling_hours": 15, "canonical_iterations": 60, "seeds": {"a": 1}}
    amended = v2.apply_to_train_config_document(document)
    assert amended["wall_clock_ceiling_hours"] == 24
    assert document["wall_clock_ceiling_hours"] == 15  # the input is not mutated
    assert amended["canonical_iterations"] == 60
    assert amended["seeds"] == {"a": 1}


def test_apply_refuses_a_document_that_is_not_at_the_v1_ceiling():
    with pytest.raises(v2.Phase9AmendmentV2Error, match="v1-amended"):
        v2.apply_to_train_config_document({"wall_clock_ceiling_hours": 12})
    with pytest.raises(v2.Phase9AmendmentV2Error, match="no 'wall_clock_ceiling_hours'"):
        v2.apply_to_train_config_document({"canonical_iterations": 60})


def test_reconciliation_reports_exactly_one_changed_field():
    document = {"wall_clock_ceiling_hours": 15, "canonical_iterations": 60}
    result = v2.reconcile_documents(document, v2.apply_to_train_config_document(document))
    assert result["only_the_wall_clock_ceiling_changed"] is True
    assert result["unchanged_field_count"] == 1
    assert result["changed_fields"] == [
        {"field": "wall_clock_ceiling_hours", "previous": 15, "amended": 24}
    ]


def test_reconciliation_catches_a_second_changed_field():
    """The negative control: a learning quantity riding along must be visible."""
    document = {"wall_clock_ceiling_hours": 15, "canonical_iterations": 60}
    smuggled = v2.apply_to_train_config_document(document)
    smuggled["canonical_iterations"] = 40
    result = v2.reconcile_documents(document, smuggled)
    assert result["only_the_wall_clock_ceiling_changed"] is False
    assert {entry["field"] for entry in result["changed_fields"]} == {
        "wall_clock_ceiling_hours",
        "canonical_iterations",
    }


# ---------------------------------------------------------------------------
# What it does not change
# ---------------------------------------------------------------------------


def test_unchanged_manifest_reads_the_live_frozen_contract():
    unchanged = v2.amendment_document()["unchanged"]
    assert unchanged["canonical_iterations"] == CANONICAL_ITERATIONS == 60
    assert unchanged["canonical_games_per_iteration"] == CANONICAL_GAMES_PER_ITERATION
    assert unchanged["epochs_per_rollout"] == EPOCHS_PER_ROLLOUT
    assert unchanged["validation_cadence_iterations"] == VALIDATION_CADENCE_ITERATIONS
    assert unchanged["archive_cadence_iterations"] == ARCHIVE_CADENCE_ITERATIONS
    assert unchanged["validation_passes"] == 12
    assert unchanged["winning_candidate_id"] == "P9-C"
    assert unchanged["learning_rate"] == pytest.approx(3e-4)
    assert unchanged["initial_kl_beta"] == pytest.approx(0.005)


def test_amendment_forbids_the_things_review_did_not_authorize():
    forbidden = " ".join(
        v2.amendment_document()["authorization"]["explicitly_not_authorized"]
    ).lower()
    for phrase in (
        "extra games",
        "extra optimizer updates",
        "additional validation",
        "additional archive members",
        "hyperparameter changes",
        "experimentation",
        "sealed final-test bank",
        "phase9_rl_contract_v1 in place",
        "phase9_operational_amendment_v1 in place",
    ):
        assert phrase in forbidden


def test_the_ceiling_is_a_maximum_not_a_training_target():
    rule = v2.amendment_document()["change"]["ceiling_rule"].lower()
    assert "maximum" in rule and "not a training target" in rule
    assert "unused allowance is never spent" in rule
    assert "incomplete" in rule


def test_game_length_is_not_claimed_as_evidence_of_strength():
    """The review's explicit framing, carried in the artifact rather than lost
    in a chat message: longer games are a runtime observation, and strength
    claims stay with validation and Agent 8's sealed final test."""
    note = v2.amendment_document()["authorization"]["game_length_interpretation"].lower()
    assert "not by itself evidence of stronger play" in note
    assert "validation" in note and "agent 8" in note


def test_the_trigger_records_the_measurement_not_a_narrative():
    trigger = v2.amendment_document()["authorization"]["trigger"]
    for token in ("324,990", "431,214", "20,425", "54,000"):
        assert token in trigger


def test_the_finding_is_labelled_operational_not_a_training_failure():
    finding = v2.amendment_document()["authorization"]["finding"].lower()
    assert "operational finding" in finding
    assert "not a training failure" in finding
    assert "no hard stop fired" in finding


# ---------------------------------------------------------------------------
# The runtime identity
# ---------------------------------------------------------------------------


def test_runtime_identity_carries_no_wall_clock_field():
    from stratego.training import phase9_trainer as pt

    identity = pt.Phase9TrainConfig.for_candidate(
        "P9-C", namespace="canonical", device="cpu", total_iterations=60
    ).identity()
    assert not any("wall_clock" in key or "ceiling" in key for key in identity)
    effect = v2.runtime_identity_is_unaffected(identity, identity)
    assert effect["unchanged"] is True
    assert effect["carries_a_wall_clock_field"] is False
    assert v2.amendment_document()["affects_trainer_runtime_identity"] is False


def test_runtime_identity_effect_detects_a_real_change():
    """The negative control for the measurement itself."""
    effect = v2.runtime_identity_is_unaffected(
        {"learning_rate": 3e-4}, {"learning_rate": 1e-4}
    )
    assert effect["unchanged"] is False
    assert effect["differing_fields"] == ["learning_rate"]
