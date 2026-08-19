"""Phase 10 Agent 5's three artifacts, checked against the frozen contract.

The evaluation itself is expensive and ran once; what these tests protect is
the record of it. Every claim below is recomputed from the artifact's own
primitives — the score from its components, the ranking from the frozen
tie-break, the eligibility from the frozen thresholds — so an edited number
fails here rather than being inherited by Agent 6.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from stratego.training.phase10_acceptance import select_winner, tie_break_key
from stratego.training.phase10_contract import (
    CANDIDATE_IDS,
    CANDIDATE_MATRIX,
    LEARNED_MIXTURE_WEIGHT,
    MATCHUP_TOKENS,
    NEUTRAL_MIXTURE_WEIGHT,
    SELECTION_SCORE_WEIGHTS,
    TIE_BREAK_ORDER,
    VALIDATION_BASIC_MIN_EWR,
    VALIDATION_RANDOM_MIN_EWR,
)
from stratego.training.phase10_seed import CANONICAL_PHASE10_SEEDS

from .phase10_frozen_digests import BANK_DIGESTS, CONTRACT_BUNDLE_DIGEST

#: The accepted Phase 9 move model, from the Phase 10 common contract.
ACCEPTED_PHASE9_CHECKPOINT_SHA256 = (
    "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
)
ACCEPTED_PHASE9_MODEL_STATE_DIGEST = (
    "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
)
VALIDATION_BANK_DIGEST = BANK_DIGESTS["validation"]

DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "phase_10_data"
ACCEPTANCE_PATH = DATA_DIRECTORY / "agent_05_acceptance.json"
CONFIG_PATH = DATA_DIRECTORY / "agent_05_frozen_selector_config.json"
RESULTS_PATH = DATA_DIRECTORY / "agent_05_candidate_results.csv"

#: 128 logical paired cases x 2 colour-paired games.
GAMES_PER_CELL = 256
#: 6 candidates x 6 matchups + the baseline arm's 5 matchups, all x 256.
TOTAL_GAMES = (6 * 6 + 5) * GAMES_PER_CELL


@pytest.fixture(scope="module")
def acceptance() -> dict:
    return json.loads(ACCEPTANCE_PATH.read_text())


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def results() -> list:
    with open(RESULTS_PATH, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def candidates(acceptance) -> dict:
    return {entry["candidate_id"]: entry for entry in acceptance["candidates"]}


# ---------------------------------------------------------------------------
# Status and upstream identity
# ---------------------------------------------------------------------------


#: `full_suite_green` is a claim about the suite that contains this test, so
#: asserting it here would be a fixed point rather than a check: a false gate
#: would fail the suite, which would keep the gate false forever. It is
#: checked against the recorded suite measurement instead, one line below.
SELF_REFERENTIAL_GATE = "full_suite_green"


def test_every_gate_agent_5_can_evidence_is_true(acceptance):
    gates = acceptance["completion_gates"]
    independent = {
        name: value for name, value in gates.items() if name != SELF_REFERENTIAL_GATE
    }
    assert independent
    assert all(independent.values()), sorted(
        name for name, value in independent.items() if not value
    )


def test_the_suite_gate_agrees_with_the_recorded_measurement(acceptance):
    """The self-referential gate, checked the only way it can be.

    The gate must say exactly what the stored suite run says — so a green
    gate over a red run, or a stale measurement, fails here.
    """
    suite = acceptance["suite"]
    assert acceptance["completion_gates"][SELF_REFERENTIAL_GATE] == (
        suite["returncode"] == 0 and suite["failed"] == 0
    )
    assert suite["command"] == ".venv/bin/python -m pytest tests -q"
    assert suite["passed"] > acceptance["suite_before"]["passed"]


def test_status_follows_from_the_recorded_gates(acceptance):
    gates = acceptance["completion_gates"]
    false_gates = sorted(name for name, value in gates.items() if not value)
    assert acceptance["false_gates"] == false_gates
    assert acceptance["gates_total"] == len(gates)
    assert acceptance["gates_true"] == sum(1 for value in gates.values() if value)
    assert acceptance["status"] == ("PASS" if not false_gates else "FAIL")


def test_the_frozen_upstream_identities_are_the_accepted_ones(acceptance):
    frozen = acceptance["frozen_inputs"]
    assert frozen["contract_bundle_digest"] == CONTRACT_BUNDLE_DIGEST
    assert frozen["phase9_checkpoint_sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert frozen["phase9_model_state_digest"] == ACCEPTED_PHASE9_MODEL_STATE_DIGEST
    assert frozen["validation_bank_digest"] == VALIDATION_BANK_DIGEST


def test_the_phase9_checkpoint_is_byte_identical_before_and_after(acceptance):
    preservation = acceptance["phase9_preservation"]
    assert preservation["unchanged"] is True
    assert preservation["before"] == preservation["after"]
    assert preservation["c1_optimizer_steps"] == 0


# ---------------------------------------------------------------------------
# Bounded scope
# ---------------------------------------------------------------------------


def test_exactly_the_six_frozen_candidates_were_evaluated(acceptance, candidates):
    assert set(candidates) == set(CANDIDATE_IDS)
    assert len(acceptance["candidates"]) == 6
    matrix = {entry["candidate_id"]: entry for entry in CANDIDATE_MATRIX}
    for candidate_id, record in candidates.items():
        assert record["utility_model"] == matrix[candidate_id]["utility_model"]
        assert record["temperature"] == matrix[candidate_id]["temperature"]


def test_nothing_was_refit_retuned_or_rerun(acceptance):
    discipline = acceptance["discipline"]
    assert discipline["utility_models_fit"] == 0
    assert discipline["candidates_added"] == 0
    assert discipline["temperature_changes"] == 0
    assert discipline["mixture_changes"] == 0
    assert discipline["rescue_reruns"] == 0
    assert discipline["c1_optimizer_steps"] == 0
    assert discipline["human_games_used"] == 0
    assert discipline["corpus_records_read"] == 0


def test_the_test_bank_stayed_sealed(acceptance):
    discipline = acceptance["discipline"]
    assert discipline["test_bank_outcome_access"] == 0
    assert discipline["test_bank_neural_inference"] == 0
    for entry in acceptance["bank_access_log"]:
        if entry["bank"] == "phase10_test_bank_v1":
            assert entry["neural"] is False
            assert entry["outcomes"] is False
    handoff = acceptance["handoff_to_agent_6"]["test_bank_unopened"]
    assert handoff["games"] == 0
    assert handoff["neural_inference"] == 0
    assert handoff["outcomes_read"] == 0


def test_every_candidate_and_the_baseline_saw_the_same_cases(acceptance):
    assert acceptance["discipline"]["validation_bank_outcome_access"] == TOTAL_GAMES
    assert acceptance["discipline"]["games_played"] == TOTAL_GAMES
    cells = acceptance["new_digests"]["cell_result_digests"]
    assert set(cells) == set(CANDIDATE_IDS)
    for candidate_cells in cells.values():
        assert set(candidate_cells) == set(MATCHUP_TOKENS)
    neutral = acceptance["new_digests"]["neutral_arm_digests"]
    assert set(neutral) == {token for token in MATCHUP_TOKENS if token != "learned_vs_neutral"}


def test_no_two_cells_share_a_result_digest(acceptance):
    """Distinct cells must be distinct games, not a reused cache."""
    digests = [
        digest
        for candidate_cells in acceptance["new_digests"]["cell_result_digests"].values()
        for digest in candidate_cells.values()
    ] + list(acceptance["new_digests"]["neutral_arm_digests"].values())
    assert len(digests) == len(set(digests)) == 6 * 6 + 5


# ---------------------------------------------------------------------------
# The learned branch verification, and its negative control
# ---------------------------------------------------------------------------


def test_the_learned_branch_was_verified_before_the_first_game(acceptance):
    verification = acceptance["learned_branch_verification"]
    assert verification["problems"] == []
    assert verification["ran_before_any_validation_game"] is True

    structural = verification["structural"]
    assert structural["branch_coin_calls_in_draw"] == 1
    assert structural["base_uniform_calls_in_draw"] == 1
    assert structural["mixture_weight_comparisons_in_draw"] == [
        "branch_uniform < NEUTRAL_MIXTURE_WEIGHT"
    ]
    assert structural["bare_mixture_literals_in_draw"] == []
    assert structural["ladder_assignment"] == "np.cumsum(p_learned)"
    assert "cumulative_learned" in structural["attributes_read_by_the_walk"]
    assert "p_mixed" not in structural["attributes_read_by_the_walk"]

    runtime = verification["runtime"]
    assert runtime["one_coin_per_draw"] is True
    assert runtime["base_uniform_only_on_the_learned_branch"] is True


def test_the_negative_control_reproduced_the_superseded_double_mixing(acceptance):
    """The control is what gives the verification its sensitivity."""
    control = acceptance["learned_branch_verification"]["negative_control"]
    assert control["shadow_ladder"] == "cumsum(p_mixed)"
    predicted = control["predicted_realization"]
    assert predicted["neutral_weight"] == pytest.approx(0.5775, abs=1e-12)
    assert predicted["learned_weight"] == pytest.approx(0.4225, abs=1e-12)
    assert control["rows"]
    for row in control["rows"]:
        assert row["defective_tv_to_p_mixed"] > 4.0 * row["production_tv_to_p_mixed"]
        assert (
            row["defective_tv_to_double_mixed_prediction"] < row["defective_tv_to_p_mixed"]
        )


# ---------------------------------------------------------------------------
# Score, eligibility and tie-break, recomputed
# ---------------------------------------------------------------------------


def test_s10_recomputes_from_its_components(candidates):
    for record in candidates.values():
        recomputed = sum(
            SELECTION_SCORE_WEIGHTS[name] * record["score"]["components"][name]
            for name in SELECTION_SCORE_WEIGHTS
        )
        assert recomputed == pytest.approx(record["s10"], abs=1e-15)
        assert record["delta_direct"] == pytest.approx(
            record["summaries"]["learned_vs_neutral"]["learned_ewr"] - 0.5, abs=1e-15
        )
        for name, token in (
            ("delta_strategic", "vs_strategic"),
            ("delta_tactical", "vs_tactical"),
            ("delta_phase8_anchor", "vs_phase8_anchor"),
        ):
            assert record[name] == pytest.approx(record["summaries"][token]["delta"], abs=1e-15)


def test_the_independent_score_check_agrees(acceptance):
    check = acceptance["independent_score_check"]
    assert check["all_agree"] is True
    assert len(check["rows"]) == 6
    for row in check["rows"]:
        assert abs(row["difference"]) <= 1e-15


def test_eligibility_applies_exactly_the_frozen_thresholds(acceptance, candidates):
    for record in candidates.values():
        guards = record["guards"]
        assert guards["random_min"] == VALIDATION_RANDOM_MIN_EWR
        assert guards["basic_min"] == VALIDATION_BASIC_MIN_EWR
        assert guards["checks"]["random_overall"] == (
            guards["random_ewr"] >= VALIDATION_RANDOM_MIN_EWR
        )
        assert guards["checks"]["basic"] == (guards["basic_ewr"] >= VALIDATION_BASIC_MIN_EWR)
        expected = (
            guards["all_pass"]
            and record["correctness_clean"]
            and not record["ineligible_reasons"]
        )
        assert record["eligible"] is expected


def test_random_and_basic_are_guards_and_never_score_components(candidates):
    assert set(SELECTION_SCORE_WEIGHTS) == {
        "delta_direct",
        "delta_strategic",
        "delta_tactical",
        "delta_phase8_anchor",
    }
    for record in candidates.values():
        assert set(record["score"]["components"]) == set(SELECTION_SCORE_WEIGHTS)


def test_the_ranking_reproduces_under_the_frozen_tie_break(acceptance, candidates):
    reproduced = select_winner(
        [
            {
                "candidate_id": record["candidate_id"],
                "eligible": record["eligible"],
                "s10": record["s10"],
                "delta_strategic": record["delta_strategic"],
                "delta_direct": record["delta_direct"],
                "normalized_family_entropy": record["normalized_family_entropy"],
                "effective_base_diversity": record["effective_base_diversity"],
            }
            for record in candidates.values()
        ]
    )
    assert reproduced["ranking"] == acceptance["selection"]["ranking"]
    assert reproduced["winner"] == acceptance["selection"]["winner"]
    assert acceptance["tie_break"]["order"] == acceptance["selection"]["ranking"]
    assert list(TIE_BREAK_ORDER) == acceptance["tie_break"]["levels"]


def test_the_winner_is_eligible_and_ranks_first(acceptance, candidates):
    winner_id = acceptance["selection"]["winner"]
    assert winner_id is not None
    assert candidates[winner_id]["eligible"] is True
    assert acceptance["selection"]["ranking"][0] == winner_id
    key = tie_break_key(
        {
            "candidate_id": winner_id,
            "s10": candidates[winner_id]["s10"],
            "delta_strategic": candidates[winner_id]["delta_strategic"],
            "delta_direct": candidates[winner_id]["delta_direct"],
            "normalized_family_entropy": candidates[winner_id]["normalized_family_entropy"],
            "effective_base_diversity": candidates[winner_id]["effective_base_diversity"],
        }
    )
    for other in acceptance["selection"]["ranking"][1:]:
        record = candidates[other]
        assert key > tie_break_key(
            {
                "candidate_id": other,
                "s10": record["s10"],
                "delta_strategic": record["delta_strategic"],
                "delta_direct": record["delta_direct"],
                "normalized_family_entropy": record["normalized_family_entropy"],
                "effective_base_diversity": record["effective_base_diversity"],
            }
        )


def test_an_ineligible_candidate_could_not_have_won(candidates):
    """The rule, stated on this run's own numbers rather than in the abstract."""
    ineligible = [record for record in candidates.values() if not record["eligible"]]
    reproduced = select_winner(
        [
            {
                "candidate_id": record["candidate_id"],
                # Force every candidate ineligible except the highest scorer.
                "eligible": False,
                "s10": record["s10"],
                "delta_strategic": record["delta_strategic"],
                "delta_direct": record["delta_direct"],
                "normalized_family_entropy": record["normalized_family_entropy"],
                "effective_base_diversity": record["effective_base_diversity"],
            }
            for record in candidates.values()
        ]
    )
    assert reproduced["winner"] is None
    assert reproduced["outcome"] == "FAIL"
    assert reproduced["no_eligible_candidate"] is True
    for record in ineligible:
        assert record["ineligible_reasons"]


# ---------------------------------------------------------------------------
# Seat-policy reconciliation
# ---------------------------------------------------------------------------

#: Derived from the frozen matchup mapping, not read off the data: the
#: selector seat is the Phase 9 checkpoint in all six matchups, the direct
#: matchup contributes a second Phase 9 seat, and each external opponent
#: holds one seat in six candidate cells plus the baseline cell.
EXPECTED_PHASE9_SEATS = 6 * GAMES_PER_CELL * 2 + 6 * 5 * GAMES_PER_CELL + 5 * GAMES_PER_CELL
EXPECTED_OPPONENT_SEATS = 6 * GAMES_PER_CELL + GAMES_PER_CELL


def test_the_expected_seat_counts_cover_every_seat_of_every_game():
    """The arithmetic the audit is checked against, checked itself."""
    assert EXPECTED_PHASE9_SEATS == 12_032
    assert EXPECTED_OPPONENT_SEATS == 1_792
    assert EXPECTED_PHASE9_SEATS + 5 * EXPECTED_OPPONENT_SEATS == 2 * TOTAL_GAMES


def test_both_seats_of_every_recorded_game_were_reconciled(acceptance):
    audit = acceptance["seat_policy_audit"]
    assert audit["problems"] == []
    assert audit["mismatches"] == 0
    assert audit["games_audited"] == TOTAL_GAMES
    assert audit["seats_audited"] == 2 * TOTAL_GAMES
    assert audit["aggregate_matches_expected"] is True


def test_the_aggregate_seat_counts_are_the_expected_ones(acceptance):
    audit = acceptance["seat_policy_audit"]
    counts = audit["aggregate_seat_counts"]
    phase9 = [token for token in counts if "phase10_eval_move_v1" in token]
    assert len(phase9) == 1
    assert counts[phase9[0]] == EXPECTED_PHASE9_SEATS
    others = {token: value for token, value in counts.items() if token != phase9[0]}
    assert len(others) == 5
    assert set(others.values()) == {EXPECTED_OPPONENT_SEATS}
    assert counts == audit["expected_seat_counts"]
    assert sum(counts.values()) == 2 * TOTAL_GAMES


def test_every_matchup_seats_the_selector_on_the_phase9_checkpoint(acceptance):
    audit = acceptance["seat_policy_audit"]
    by_matchup: dict = {}
    for entry in audit["per_matchup_seats"]:
        by_matchup.setdefault(entry["matchup"], []).append(entry)
    assert set(by_matchup) == set(MATCHUP_TOKENS)
    for matchup, entries in by_matchup.items():
        games = audit["games_by_matchup"][matchup]
        selector = [entry for entry in entries if entry["role"] == "selector"]
        opposing = [entry for entry in entries if entry["role"] == "opposing"]
        assert len(selector) == len(opposing) == 1
        assert "phase10_eval_move_v1" in selector[0]["policy_token"]
        for entry in (selector[0], opposing[0]):
            # The frozen colour pairing puts the selector on Red in game 0 and
            # Blue in game 1, so every seat is exactly half and half.
            assert entry["red"] == entry["blue"] == games // 2
            assert entry["total"] == games
        if matchup == "learned_vs_neutral":
            assert opposing[0]["policy_token"] == selector[0]["policy_token"]
        else:
            assert opposing[0]["policy_token"] != selector[0]["policy_token"]


def test_the_checkpoint_behind_each_neural_seat_was_controlled_both_ways(acceptance):
    """A token names a policy; the control is what binds it to weights."""
    binding = acceptance["seat_policy_audit"]["weights_binding"]
    assert {entry["matchup"] for entry in binding} == {
        "learned_vs_neutral",
        "vs_phase8_anchor",
    }
    for entry in binding:
        assert entry["sampled_games"] > 0
        assert entry["correct_owner_reproduces"] == entry["sampled_games"]
        assert entry["swapped_owner_changes_the_game"] == entry["sampled_games"]
        assert entry["bound_checkpoint"] != entry["swapped_checkpoint"]


def test_the_audit_reran_no_scheduled_game_and_changed_no_selection(acceptance):
    audit = acceptance["seat_policy_audit"]
    assert audit["scheduled_games_rerun"] == 0
    assert audit["selection_changed"] is False
    assert acceptance["discipline"]["games_played"] == TOTAL_GAMES


def test_a_replayed_work_unit_reproduced_its_recorded_games(acceptance):
    """Sharding is not an input to a result."""
    replay = acceptance["unit_replay"]
    assert replay["digest_identical"] is True
    assert replay["every_field_identical"] is True
    assert replay["fresh_process"] is True
    assert replay["replay_workers"] != replay["recorded_workers"]
    assert replay["games"] > 0


# ---------------------------------------------------------------------------
# Correctness and the report-only diagnostic
# ---------------------------------------------------------------------------


def test_every_zero_tolerance_counter_is_zero(acceptance, candidates):
    assert acceptance["discipline"]["inference_failures"] == 0
    for record in candidates.values():
        assert record["correctness_clean"] is True
        assert all(value == 0 for value in record["safety"].values())


def test_the_phase9_landing_diagnostic_is_recorded_at_the_required_granularity(acceptance):
    diagnostic = acceptance["landing_diagnostic"]
    assert diagnostic["granularity"] == "candidate x arm x matchup x bank"
    assert diagnostic["use"] == "report_only"
    assert diagnostic["gate"] is False

    rows = diagnostic["rows"]
    keys = {(row["candidate_id"], row["arm"], row["matchup"], row["bank"]) for row in rows}
    assert len(keys) == len(rows) == 6 * 6 + 5
    for row in rows:
        assert row["bank"] == "phase10_validation_bank_v1"
        assert row["games"] == GAMES_PER_CELL
        assert 0 <= row["landings"] <= row["games"]
        assert row["landing_rate"] == pytest.approx(row["landings"] / row["games"], abs=1e-12)

    # An own-side draw depends on the case, the colour and the candidate and
    # never on the opponent, so the count is constant within a candidate.
    for candidate_id in (*CANDIDATE_IDS, "neutral_v1"):
        counts = {row["landings"] for row in rows if row["candidate_id"] == candidate_id}
        assert len(counts) == 1


def test_no_gate_or_score_reads_the_landing_diagnostic(acceptance):
    """The diagnostic must not appear anywhere a decision was made."""
    decision_surfaces = json.dumps(
        {
            "gates": acceptance["completion_gates"],
            "selection": acceptance["selection"],
            "tie_break": acceptance["tie_break"],
            "score_check": acceptance["independent_score_check"],
            "guards": [record["guards"] for record in acceptance["candidates"]],
            "reasons": [record["ineligible_reasons"] for record in acceptance["candidates"]],
        }
    )
    for term in ("landing", "fingerprint", "phase9_isolation"):
        assert term not in decision_surfaces


# ---------------------------------------------------------------------------
# The frozen selector configuration
# ---------------------------------------------------------------------------


def test_the_frozen_config_names_the_winner_and_the_frozen_mixture(config, acceptance):
    assert config["selector_config_version"] == "phase10_selector_config_v1"
    assert config["status"] == "SELECTED"
    assert config["winner"]["candidate_id"] == acceptance["selection"]["winner"]
    assert config["mixture"]["neutral_weight"] == NEUTRAL_MIXTURE_WEIGHT
    assert config["mixture"]["learned_weight"] == LEARNED_MIXTURE_WEIGHT
    assert config["mixture"]["applied"] == "exactly once, at the branch decision"


def test_the_frozen_config_carries_every_required_field(config):
    for field in (
        "winner", "utility", "mixture", "versions", "phase7_identity",
        "phase9_identity", "phase10_seeds", "validation_identity", "score",
        "diversity", "distribution_digests", "train_split_production_digests",
    ):
        assert field in config, field
    assert config["phase10_seeds"] == dict(CANONICAL_PHASE10_SEEDS)
    assert config["validation_identity"]["bank_digest"] == VALIDATION_BANK_DIGEST
    assert config["phase9_identity"]["checkpoint_sha256"] == ACCEPTED_PHASE9_CHECKPOINT_SHA256
    assert config["score"]["weights"] == dict(SELECTION_SCORE_WEIGHTS)
    assert config["score"]["tie_break_order"] == list(TIE_BREAK_ORDER)


def test_the_config_and_the_utility_stay_separate_artifacts(config):
    assert config["utility"]["separate_artifact_from_this_config"] is True
    assert config["utility"]["refit_by_agent_5"] is False
    assert config["utility"]["artifact"] == "checkpoints/phase10/setup_utility_v1.json"
    assert config["c1_checkpoint_created_or_altered"] is False
    # The config names the coefficients by digest; it does not embed them.
    assert "family_offsets_raw" not in json.dumps(config)


def test_the_config_score_matches_the_winners_recorded_score(config, acceptance, candidates):
    winner = candidates[acceptance["selection"]["winner"]]
    assert config["score"]["s10"] == pytest.approx(winner["s10"], abs=1e-15)
    assert config["score"]["components"] == winner["score"]["components"]


# ---------------------------------------------------------------------------
# The candidate-results CSV
# ---------------------------------------------------------------------------


def test_the_csv_carries_all_six_candidates_with_their_verdicts(results, candidates):
    assert [row["candidate_id"] for row in results] == sorted(CANDIDATE_IDS)
    for row in results:
        record = candidates[row["candidate_id"]]
        assert (row["eligible"] == "True") is record["eligible"]
        if not record["eligible"]:
            assert row["ineligible_reasons"]
        assert float(row["s10"]) == pytest.approx(record["s10"], abs=1e-9)


def test_an_unrun_candidate_would_carry_no_fabricated_score(results, candidates):
    """A blank score is the contract; a zero would be a fabricated one."""
    for row in results:
        record = candidates[row["candidate_id"]]
        if record.get("summaries") is None:
            assert row["s10"] == ""
            assert row["direct_ewr"] == ""
        else:
            assert row["s10"] != ""


def test_the_csv_marks_exactly_one_winner(results, acceptance):
    winners = [row["candidate_id"] for row in results if row["winner"] == "True"]
    assert winners == [acceptance["selection"]["winner"]]
    ranks = sorted(int(row["rank"]) for row in results if row["rank"])
    assert ranks == list(range(1, len(ranks) + 1))
