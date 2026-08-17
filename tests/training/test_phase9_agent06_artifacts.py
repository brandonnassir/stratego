"""Phase 9 Agent 6: the published pilot-selection artifacts stay honest.

These tests read the reports rather than recompute them. Their job is to stop
a published artifact from drifting away from the frozen contract it claims to
satisfy — a selection whose score does not recompute from its own recorded
EWRs, a winner chosen off the frozen tie-break, a frozen train config whose
digest no longer hashes its own document, an iteration-4 guard silently
promoted to a binding veto, or a "final-test untouched" claim with a test-bank
matchup in the evidence.

The artifacts exist only after `scripts/run_phase9_agent06.py` has run, so
every test skips cleanly when they are absent rather than failing a fresh
checkout.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from stratego.training.phase9_contract import (
    ARCHIVE_CADENCE_ITERATIONS,
    CANONICAL_GAMES_PER_ITERATION,
    CANONICAL_ITERATIONS,
    CANONICAL_WALL_CLOCK_CEILING_HOURS,
    EPOCHS_PER_ROLLOUT,
    EXPECTED_PHASE8_CHECKPOINT_SHA256,
    MINIBATCH_SIZE,
    PILOT_BUCKET_COUNTS,
    PILOT_CANDIDATES,
    PILOT_GAMES_PER_ITERATION,
    PILOT_HARD_VETOES,
    PILOT_ITERATIONS,
    TEST_BANK_VERSION,
    VALIDATION_BANK_VERSION,
    VALIDATION_CADENCE_ITERATIONS,
    VALIDATION_REGRESSION_GUARDS,
    VALIDATION_SCORE_WEIGHTS,
    VALIDATION_TIE_BREAK,
    active_historical_window,
    contract_digest,
    iter_scheduled_games,
    validation_score,
)
from stratego.training.phase9_seed import CANONICAL_PHASE9_SEEDS

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "phase_9_data"
SELECTION = DATA_DIRECTORY / "agent_06_pilot_selection.json"
RUNS = DATA_DIRECTORY / "agent_06_pilot_runs.csv"
CONFIG = DATA_DIRECTORY / "agent_06_frozen_train_config.json"

ACCEPTED_CONTRACT_DIGEST = (
    "ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34"
)

#: A test that runs *inside* the suite cannot soundly assert that the suite
#: passed. That gate is established by `--record-final-suite`.
SELF_REFERENTIAL_GATE = "full_suite_green"

#: The wall-clock ceiling gate reports a measured outcome, not an artifact
#: property: a projection honestly above the frozen 12-hour ceiling is the
#: contract's own `BLOCKED — CANONICAL WALL-CLOCK CONTRACT REQUIRES REVIEW`
#: verdict, and the suite must stay green while that verdict stands for
#: review. Its *consistency* with the recorded numbers is asserted instead.
OUTCOME_GATES = ("canonical_projection_within_ceiling",)

CANDIDATE_IDS = tuple(entry["candidate_id"] for entry in PILOT_CANDIDATES)


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.name} has not been produced yet")
    return json.loads(path.read_text())


@pytest.fixture
def selection():
    return _load(SELECTION)


@pytest.fixture
def config():
    return _load(CONFIG)


@pytest.fixture
def runs_rows():
    if not RUNS.exists():
        pytest.skip(f"{RUNS.name} has not been produced yet")
    with RUNS.open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Prerequisites and identity
# ---------------------------------------------------------------------------


def test_selection_pins_the_accepted_upstream(selection):
    prerequisites = selection["prerequisites"]
    assert prerequisites["contract_digest"] == ACCEPTED_CONTRACT_DIGEST
    assert prerequisites["phase8_checkpoint_sha256"] == EXPECTED_PHASE8_CHECKPOINT_SHA256
    assert all(
        entry["status"] == "PASS" for entry in prerequisites["acceptances"].values()
    )
    assert contract_digest() == prerequisites["contract_digest"]


def test_selection_verified_corpus_and_mounted_storage(selection):
    assert selection["corpus"]["identity_matches"]
    assert selection["corpus"]["resolver"].endswith("default_corpus_root()")
    storage = selection["storage"]
    assert storage["on_external_volume"]
    assert storage["is_mount_point"]
    assert storage["write_probe_ok"]
    assert storage["resolver"].endswith("default_rollout_root()")


def test_topology_is_agent5_validated_and_not_retuned(selection):
    assert selection["topology"] == {
        "workers": 6,
        "prefetch": 2,
        "record_cache_size": 48,
    }


# ---------------------------------------------------------------------------
# The six candidates
# ---------------------------------------------------------------------------


def test_exactly_the_six_frozen_candidates_ran(selection):
    assert sorted(selection["candidates"]) == sorted(CANDIDATE_IDS)
    assert len(selection["candidates"]) == 6


def test_every_completed_candidate_ran_the_full_frozen_budget(selection):
    completed = {
        candidate_id: entry
        for candidate_id, entry in selection["candidates"].items()
        if entry["status"] == "COMPLETE"
    }
    assert completed, "no candidate completed"
    for entry in completed.values():
        totals = entry["totals"]
        assert totals["iterations_completed"] == PILOT_ITERATIONS
        assert totals["games"] == PILOT_ITERATIONS * PILOT_GAMES_PER_ITERATION


def test_all_candidates_started_from_one_identical_checkpoint(selection):
    digests = {
        entry["start_state_digest"]
        for entry in selection["candidates"].values()
        if entry["start_state_digest"] is not None
    }
    assert len(digests) == 1


def test_every_completed_candidate_validated_at_4_and_8_only(selection):
    for entry in selection["candidates"].values():
        if entry["status"] != "COMPLETE":
            continue
        assert sorted(entry["validation_scores"]) == ["4", "8"]


def test_iteration4_guards_are_diagnostic_and_iteration8_binds(selection):
    for entry in selection["candidates"].values():
        for iteration, record in entry["validation_scores"].items():
            binding = record["guards"]["binding"]
            if iteration == "8":
                assert binding == "final"
            else:
                assert binding == "intermediate_diagnostic"


def test_h005_archives_are_candidate_local_and_verified(selection):
    for candidate_id, entry in selection["candidates"].items():
        if entry["status"] != "COMPLETE":
            continue
        member = entry["archive_member"]
        namespace = member["namespace"]
        assert member["qualified_identity"] == f"{namespace}|H005"
        assert member["local_identity"] == "H005"
        assert len(member["checkpoint_sha256"]) == 64
        verifications = entry["historical_verification"]
        assert verifications, f"{candidate_id} recorded no historical verification"
        for verification in verifications:
            assert verification["all_verified"]
            if verification["iteration"] >= 6:
                assert verification["h005_checkpoint_sha256"] == member["checkpoint_sha256"]
                assert verification["verified"]["H005"]["failed"] == 0


def test_h005_reenumeration_matches_the_live_schedule(selection):
    """The recorded per-iteration H005 counts equal a fresh re-enumeration."""
    for candidate in PILOT_CANDIDATES:
        namespace = candidate["namespace"]
        recorded = selection["h005_reenumeration"][namespace]["iterations"]
        for iteration in (6, 7, 8):
            counts: dict[str, int] = {}
            for game in iter_scheduled_games(namespace, iteration):
                if game["bucket"] == "historical":
                    key = game["opponent"]["identity"]
                    counts[key] = counts.get(key, 0) + 1
            entry = recorded[str(iteration)]
            assert entry["opponent_counts"] == counts
            assert entry["active_window"] == list(active_historical_window(iteration))
            assert sum(counts.values()) == PILOT_BUCKET_COUNTS["historical"]


def test_h005_verification_covered_every_scheduled_h005_game(selection):
    for candidate_id, entry in selection["candidates"].items():
        if entry["status"] != "COMPLETE":
            continue
        namespace = next(
            item["namespace"]
            for item in PILOT_CANDIDATES
            if item["candidate_id"] == candidate_id
        )
        recorded = {
            verification["iteration"]: verification
            for verification in entry["historical_verification"]
        }
        for iteration in (6, 7, 8):
            expected = selection["h005_reenumeration"][namespace]["iterations"][
                str(iteration)
            ]["opponent_counts"].get("H005", 0)
            assert recorded[iteration]["verified"]["H005"]["games"] == expected


# ---------------------------------------------------------------------------
# Vetoes
# ---------------------------------------------------------------------------


def test_veto_evaluation_covers_exactly_the_frozen_vetoes(selection):
    for evaluation in selection["veto_evaluation"].values():
        assert evaluation["covers_exactly_frozen_vetoes"]
        assert sorted(evaluation["evaluation"]) == sorted(PILOT_HARD_VETOES)


def test_no_surviving_candidate_breaches_any_frozen_veto(selection):
    for candidate_id, entry in selection["candidates"].items():
        if entry["status"] != "COMPLETE":
            continue
        evaluation = selection["veto_evaluation"][candidate_id]["evaluation"]
        for name, record in evaluation.items():
            assert not record["breached"], f"{candidate_id} breaches {name}"
        assert entry["counters"].get("non_finite_losses", 0) == 0
        assert entry["counters"].get("behavior_identity_mismatches", 0) == 0
        assert entry["counters"].get("checkpoint_errors", 0) == 0


def test_vetoed_candidates_are_excluded_not_rescued(selection):
    for candidate_id, entry in selection["candidates"].items():
        if entry["status"] == "VETOED":
            assert entry["veto"] is not None
            assert candidate_id != selection["selection"].get("winner")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_selection_scores_recompute_from_their_own_ewrs(selection):
    for entry in selection["candidates"].values():
        for record in entry["validation_scores"].values():
            ewrs = record["effective_win_rates"]
            recomputed = validation_score(
                ewrs["strategic_rule_based"],
                ewrs["tactical_rule_based"],
                ewrs["phase8_anchor"],
            )
            assert abs(recomputed - record["selection_score"]) < 1e-12


def test_winner_has_the_strictly_best_final_score_or_tie_break(selection):
    winner_id = selection["selection"]["winner"]
    if winner_id is None:
        pytest.skip("selection recorded no winner (BLOCKED run)")
    assert selection["selection"]["unique"]
    winner_final = selection["candidates"][winner_id]["validation_scores"]["8"]
    for candidate_id, entry in selection["candidates"].items():
        if candidate_id == winner_id or entry["status"] != "COMPLETE":
            continue
        final = entry["validation_scores"].get("8")
        if final is None:
            continue
        guards = final["guards"]
        if not (guards["random_pass"] and guards["basic_pass"]):
            continue
        assert final["selection_score"] <= winner_final["selection_score"]
    assert selection["selection"]["tie_break"] == list(VALIDATION_TIE_BREAK)


def test_score_weights_and_guards_are_the_frozen_ones(selection):
    assert selection["candidates"], "no candidates recorded"
    any_entry = next(iter(selection["candidates"].values()))
    final = any_entry["validation_scores"].get("8") or next(
        iter(any_entry["validation_scores"].values())
    )
    assert final["guards"]["random_min"] == VALIDATION_REGRESSION_GUARDS[
        "random_legal_ewr_min"
    ]
    assert final["guards"]["basic_min"] == VALIDATION_REGRESSION_GUARDS[
        "basic_heuristic_ewr_min"
    ]
    assert VALIDATION_SCORE_WEIGHTS["strategic_rule_based"] == 0.45
    assert VALIDATION_SCORE_WEIGHTS["tactical_rule_based"] == 0.35
    assert VALIDATION_SCORE_WEIGHTS["phase8_anchor"] == 0.20


def test_stress_results_are_report_only_and_not_score_components(selection):
    """Stress EWRs exist for evidence; the score recomputes without them."""
    for entry in selection["candidates"].values():
        final = entry["validation_scores"].get("8")
        if final is None:
            continue
        ewrs = final["effective_win_rates"]
        assert set(ewrs) == {
            "random_legal",
            "basic_heuristic",
            "tactical_rule_based",
            "strategic_rule_based",
            "phase8_anchor",
        }


# ---------------------------------------------------------------------------
# The CSV
# ---------------------------------------------------------------------------


def test_csv_has_a_row_per_candidate_per_validation_checkpoint(runs_rows):
    seen = {(row["candidate_id"], row["iteration"]) for row in runs_rows}
    for candidate_id in CANDIDATE_IDS:
        for iteration in ("4", "8"):
            assert (candidate_id, iteration) in seen


def test_csv_scores_recompute_from_csv_ewrs(runs_rows):
    checked = 0
    for row in runs_rows:
        if not row["selection_score"]:
            continue
        recomputed = validation_score(
            float(row["ewr_strategic"]), float(row["ewr_tactical"]), float(row["ewr_anchor"])
        )
        assert abs(recomputed - float(row["selection_score"])) < 1e-9
        checked += 1
    assert checked > 0


def test_csv_agrees_with_the_selection_artifact(selection, runs_rows):
    for row in runs_rows:
        if not row["selection_score"]:
            continue
        record = selection["candidates"][row["candidate_id"]]["validation_scores"][
            row["iteration"]
        ]
        assert abs(record["selection_score"] - float(row["selection_score"])) < 1e-12
        assert record["checkpoint_sha256"] == row["checkpoint_sha256"]


# ---------------------------------------------------------------------------
# The frozen train config
# ---------------------------------------------------------------------------


def test_frozen_config_digest_hashes_its_own_document(config):
    document = config["config"]
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    assert (
        hashlib.sha256(canonical.encode()).hexdigest()
        == config["train_config_document_digest"]
    )


def test_frozen_config_carries_the_winner_and_frozen_constants(config, selection):
    document = config["config"]
    winner_id = selection["selection"]["winner"]
    if winner_id is None:
        pytest.skip("selection recorded no winner (BLOCKED run)")
    winner = next(
        entry for entry in PILOT_CANDIDATES if entry["candidate_id"] == winner_id
    )
    assert document["winning_candidate_id"] == winner_id
    assert document["learning_rate"] == winner["learning_rate"]
    assert document["initial_kl_beta"] == winner["initial_kl_beta"]
    assert document["batch_size"] == MINIBATCH_SIZE
    assert document["epochs_per_rollout"] == EPOCHS_PER_ROLLOUT
    assert document["canonical_iterations"] == CANONICAL_ITERATIONS
    assert document["canonical_games_per_iteration"] == CANONICAL_GAMES_PER_ITERATION
    assert document["validation_cadence_iterations"] == VALIDATION_CADENCE_ITERATIONS
    assert document["archive_cadence_iterations"] == ARCHIVE_CADENCE_ITERATIONS
    assert document["start"]["checkpoint_sha256"] == EXPECTED_PHASE8_CHECKPOINT_SHA256
    assert document["seeds"] == {
        key: value for key, value in CANONICAL_PHASE9_SEEDS.items()
    }


def test_frozen_config_runtime_identity_matches_a_fresh_construction(config, selection):
    from stratego.training.phase9_trainer import Phase9TrainConfig

    winner_id = selection["selection"]["winner"]
    if winner_id is None:
        pytest.skip("selection recorded no winner (BLOCKED run)")
    runtime = Phase9TrainConfig.for_candidate(
        winner_id,
        namespace="canonical",
        device="mps",
        total_iterations=CANONICAL_ITERATIONS,
    )
    assert runtime.digest() == config["trainer_runtime_identity_digest"]
    assert runtime.identity() == config["trainer_runtime_identity"]


def test_frozen_config_labels_both_digest_namespaces(config):
    assert config["train_config_document_digest"] != config[
        "trainer_runtime_identity_digest"
    ]
    assert "namespace" in config["digest_namespace_rule"]
    reconciliation = config["reconciliation"]
    assert reconciliation["bridged_fields"]
    for bridge in reconciliation["bridged_fields"]:
        assert bridge["equal"], f"bridge {bridge['document_field']} disagrees"


def test_handoff_carries_no_pilot_checkpoint(config):
    handoff = config["handoff_to_agent_7"]
    assert handoff["no_pilot_checkpoint_handed_forward"] is True
    assert handoff["fresh_start_checkpoint_sha256"] == EXPECTED_PHASE8_CHECKPOINT_SHA256
    assert handoff["canonical_budget"]["iterations"] == CANONICAL_ITERATIONS
    assert handoff["canonical_budget"]["games_per_iteration"] == CANONICAL_GAMES_PER_ITERATION


# ---------------------------------------------------------------------------
# Sealing and the projection
# ---------------------------------------------------------------------------


def test_final_test_bank_was_never_played(selection):
    instrumentation = selection["access_instrumentation"]
    assert instrumentation["final_test_neural_games"] == 0
    assert instrumentation["final_test_neural_checkpoint_loads"] == 0
    for entry in instrumentation["log"]:
        assert entry["bank_version"] == VALIDATION_BANK_VERSION
        assert entry["bank_version"] != TEST_BANK_VERSION
        assert entry["test_bank_games_played"] == 0


def test_projection_is_measured_and_names_the_frozen_ceiling(selection):
    projection = selection["canonical_projection"]
    assert projection["ceiling_hours"] == CANONICAL_WALL_CLOCK_CEILING_HOURS
    basis = projection["measured_basis"]
    assert basis["pilot_iterations"] == PILOT_ITERATIONS
    assert basis["collection_games_per_second"] > 0
    assert basis["training_examples_per_second"] > 0
    peak = projection["projection_peak_decisions"]
    assert peak["iterations"] == CANONICAL_ITERATIONS
    assert peak["validation_passes"] == CANONICAL_ITERATIONS // VALIDATION_CADENCE_ITERATIONS
    assert peak["projected_total_seconds"] > 0
    verdict = projection["verdict"]
    fits = peak["projected_total_seconds"] <= projection["ceiling_seconds"]
    if fits:
        assert verdict == "WITHIN_CEILING"
    else:
        assert verdict.startswith("BLOCKED")


def test_gates_are_internally_consistent(selection):
    gates = selection["gates"]
    assert selection["passed"] == sum(1 for value in gates.values() if value)
    assert selection["total"] == len(gates)
    assert selection["all_passed"] == all(gates.values())
    for name, value in gates.items():
        if name == SELF_REFERENTIAL_GATE or name in OUTCOME_GATES:
            continue
        assert value, f"gate {name} is not true"


def test_ceiling_gate_agrees_with_the_recorded_projection(selection):
    projection = selection["canonical_projection"]
    fits = (
        projection["projection_peak_decisions"]["projected_total_seconds"]
        <= projection["ceiling_seconds"]
    )
    assert selection["gates"]["canonical_projection_within_ceiling"] == fits
    if not fits:
        assert selection["status"].startswith("BLOCKED — CANONICAL WALL-CLOCK")


def test_full_suite_green_is_recorded_by_the_two_pass_convergence(selection):
    if not selection.get("covers_agent_06_artifact_tests"):
        pytest.skip("--record-final-suite has not converged yet")
    assert selection["gates"][SELF_REFERENTIAL_GATE]
