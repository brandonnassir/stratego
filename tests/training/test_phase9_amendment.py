"""Phase 9: `phase9_operational_amendment_v1` stays inside its reviewed scope.

The amendment exists because the reviewing chat raised one operational
number. These tests exist because an amendment is exactly the kind of object
that quietly grows: the danger is not that the ceiling moved, it is that
something else moves later under the same authorization, or that the base
contract gets "tidied" to match and silently invalidates every sealed rollout
and checkpoint that carries its digest.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from stratego.training import phase9_amendment as amendment
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


# ---------------------------------------------------------------------------
# The base contract is not touched
# ---------------------------------------------------------------------------


def test_base_contract_digest_is_unchanged():
    """The digest stamped into every sealed rollout and checkpoint still holds."""
    assert contract_digest() == ACCEPTED_CONTRACT_DIGEST
    assert amendment.AMENDED_CONTRACT_DIGEST == ACCEPTED_CONTRACT_DIGEST


def test_historical_ceiling_is_preserved_in_the_contract():
    assert CANONICAL_WALL_CLOCK_CEILING_HOURS == 12
    assert amendment.HISTORICAL_CEILING_HOURS == 12
    assert amendment.HISTORICAL_CEILING_SECONDS == 43_200


def test_verify_base_contract_untouched_reports_no_problems():
    assert amendment.verify_base_contract_untouched() == []


def test_amendment_declares_the_reviewed_ceiling():
    assert amendment.AMENDED_CEILING_HOURS == 15
    assert amendment.AMENDED_CEILING_SECONDS == 54_000
    assert amendment.amended_ceiling_seconds() == 54_000
    assert amendment.AMENDED_CEILING_SECONDS > amendment.HISTORICAL_CEILING_SECONDS


# ---------------------------------------------------------------------------
# The amendment document
# ---------------------------------------------------------------------------


def test_amendment_digest_hashes_its_own_document():
    document = amendment.amendment_document()
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == amendment.amendment_digest()


def test_amendment_records_no_in_place_edit():
    document = amendment.amendment_document()
    assert document["amends"]["in_place_edit"] is False
    assert document["amends"]["contract_digest"] == ACCEPTED_CONTRACT_DIGEST
    assert document["change"]["historical_value_preserved"] is True


def test_amendment_changes_exactly_one_field():
    change = amendment.amendment_document()["change"]
    assert change["field"] == "wall_clock_ceiling_hours"
    assert change["from_seconds"] == 43_200
    assert change["to_seconds"] == 54_000


def test_unchanged_manifest_reads_the_live_frozen_contract():
    """A frozen quantity edited elsewhere cannot hide behind this paperwork."""
    manifest = amendment.unchanged_manifest()
    assert manifest["canonical_iterations"] == CANONICAL_ITERATIONS == 60
    assert manifest["canonical_games_per_iteration"] == CANONICAL_GAMES_PER_ITERATION == 2048
    assert manifest["epochs_per_rollout"] == EPOCHS_PER_ROLLOUT == 2
    assert manifest["validation_cadence_iterations"] == VALIDATION_CADENCE_ITERATIONS == 5
    assert manifest["validation_passes"] == 12
    assert manifest["archive_cadence_iterations"] == ARCHIVE_CADENCE_ITERATIONS == 5
    assert manifest["winning_candidate_id"] == "P9-C"
    assert manifest["learning_rate"] == 3e-4
    assert manifest["initial_kl_beta"] == 0.005


def test_amendment_forbids_the_things_review_did_not_authorize():
    forbidden = amendment.AMENDMENT_AUTHORIZATION["explicitly_not_authorized"]
    joined = " ".join(forbidden).lower()
    for phrase in (
        "rerunning",
        "additional pilot training",
        "60 iterations",
        "2,048 games",
        "validation passes",
        "final-test",
        "in place",
    ):
        assert phrase in joined


# ---------------------------------------------------------------------------
# Applying it to a train-config document
# ---------------------------------------------------------------------------


def _document(ceiling: int = 12) -> dict:
    return {
        "learning_rate": 3e-4,
        "initial_kl_beta": 0.005,
        "canonical_iterations": 60,
        "wall_clock_ceiling_hours": ceiling,
        "seeds": {"phase9_master": 2026081601},
    }


def test_apply_changes_only_the_ceiling():
    original = _document()
    amended = amendment.apply_to_train_config_document(original)
    assert amended["wall_clock_ceiling_hours"] == 15
    assert original["wall_clock_ceiling_hours"] == 12, "the original must not mutate"
    for key in original:
        if key != "wall_clock_ceiling_hours":
            assert amended[key] == original[key]


def test_reconciliation_reports_exactly_one_changed_field():
    original = _document()
    amended = amendment.apply_to_train_config_document(original)
    reconciliation = amendment.reconcile_documents(original, amended)
    assert reconciliation["only_the_wall_clock_ceiling_changed"]
    assert reconciliation["changed_fields"] == [
        {"field": "wall_clock_ceiling_hours", "original": 12, "amended": 15}
    ]
    assert reconciliation["unchanged_field_count"] == len(original) - 1


def test_reconciliation_catches_a_second_changed_field():
    """The negative control: a smuggled change must not read as compliant."""
    original = _document()
    amended = amendment.apply_to_train_config_document(original)
    amended["canonical_iterations"] = 40
    reconciliation = amendment.reconcile_documents(original, amended)
    assert not reconciliation["only_the_wall_clock_ceiling_changed"]
    assert len(reconciliation["changed_fields"]) == 2


def test_apply_refuses_a_document_that_is_not_at_the_historical_ceiling():
    with pytest.raises(amendment.Phase9AmendmentError):
        amendment.apply_to_train_config_document(_document(ceiling=15))
    with pytest.raises(amendment.Phase9AmendmentError):
        amendment.apply_to_train_config_document({"learning_rate": 3e-4})


# ---------------------------------------------------------------------------
# The runtime identity
# ---------------------------------------------------------------------------


def test_runtime_identity_carries_no_wall_clock_field():
    """Why the amendment provably cannot move the runtime identity digest."""
    from stratego.training.phase9_trainer import Phase9TrainConfig

    identity = Phase9TrainConfig.for_candidate(
        "P9-C", namespace="canonical", device="mps", total_iterations=60
    ).identity()
    assert not any("wall_clock" in key or "ceiling" in key for key in identity)
    effect = amendment.runtime_identity_is_unaffected(identity, identity)
    assert effect["unchanged"]
    assert effect["differing_fields"] == []
    assert not effect["carries_a_wall_clock_field"]


def test_runtime_identity_effect_detects_a_real_change():
    """Negative control: the measurement is not hard-wired to say 'unchanged'."""
    before = {"learning_rate": 3e-4, "total_iterations": 60}
    after = {"learning_rate": 6e-4, "total_iterations": 60}
    effect = amendment.runtime_identity_is_unaffected(before, after)
    assert not effect["unchanged"]
    assert effect["differing_fields"] == ["learning_rate"]


def test_canonical_run_is_defined_by_namespace_and_iterations_not_scope():
    """The legacy scope token stays; the canonical run is defined elsewhere."""
    from stratego.training.phase9_trainer import Phase9TrainConfig

    identity = Phase9TrainConfig.for_candidate(
        "P9-C", namespace="canonical", device="mps", total_iterations=60
    ).identity()
    assert identity["scope"] == "pilot_candidate"
    assert identity["namespace"] == "canonical"
    assert identity["total_iterations"] == 60
    assert identity["candidate_id"] == "P9-C"
    assert identity["learning_rate"] == 3e-4
    assert identity["initial_kl_beta"] == 0.005
