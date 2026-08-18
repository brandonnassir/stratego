"""Regression: the frozen Phase 10 contract, its digests and its invariants.

The eight documents were serialized and hashed by Agent 1 before any
Phase 10 outcome game existed. The digests below are the freeze itself: an
edit anywhere in `phase10_contract` that changes a frozen decision fails
here instead of quietly redefining the experiment. A legitimate change is a
new contract version after review, never an updated constant in this file.
"""

import json

from stratego.setups.contracts import (
    LIBRARY_JSONL_PATH,
    TEST_TOTAL,
    TRAIN_TOTAL,
    VALIDATION_TOTAL,
)
from stratego.setups.families import FAMILY_IDS
from stratego.setups.library import (
    entry_metadata_digest,
    library_content_digest,
    read_library_jsonl,
)
from stratego.training import phase10_contract as pc

class TestUpstreamIdentities:
    def test_accepted_phase9_identity_is_the_accepted_one(self):
        assert pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256 == (
            "dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea"
        )
        assert pc.ACCEPTED_PHASE9_MODEL_STATE_DIGEST == (
            "f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd"
        )
        assert pc.ACCEPTED_PHASE9_PARAMETERS == 863_959

    def test_phase7_library_digests_match_the_live_library(self):
        entries = read_library_jsonl(LIBRARY_JSONL_PATH)
        assert library_content_digest(entries) == pc.PHASE7_LIBRARY_CONTENT_DIGEST
        assert entry_metadata_digest(entries) == pc.PHASE7_LIBRARY_METADATA_DIGEST

    def test_split_sizes_are_the_frozen_ones(self):
        assert (TRAIN_TOTAL, VALIDATION_TOTAL, TEST_TOTAL) == (6400, 800, 800)


class TestCandidateMatrix:
    def test_exactly_six_candidates(self):
        assert pc.CANDIDATE_COUNT == 6
        assert pc.CANDIDATE_IDS == ("P10-A", "P10-B", "P10-C", "P10-D", "P10-E", "P10-F")

    def test_the_frozen_model_and_temperature_of_every_candidate(self):
        assert [
            (entry["candidate_id"], entry["utility_model"], entry["temperature"])
            for entry in pc.CANDIDATE_MATRIX
        ] == [
            ("P10-A", "model_F", 0.75),
            ("P10-B", "model_F", 1.25),
            ("P10-C", "model_F", 2.00),
            ("P10-D", "model_T", 0.75),
            ("P10-E", "model_T", 1.25),
            ("P10-F", "model_T", 2.00),
        ]

    def test_the_baseline_is_not_a_candidate(self):
        assert pc.BASELINE_SELECTOR_ID not in pc.CANDIDATE_IDS

    def test_every_candidate_shares_the_frozen_mixture(self):
        assert pc.NEUTRAL_MIXTURE_WEIGHT == 0.35
        assert pc.LEARNED_MIXTURE_WEIGHT == 0.65


class TestSelectorContract:
    def test_allowed_inputs_are_exactly_the_six(self):
        assert pc.ALLOWED_SELECTOR_INPUTS == (
            "own color",
            "requested Phase 7 split",
            "candidate base's own family",
            "candidate base's own trait vector",
            "selector identity",
            "selector seed",
        )

    def test_opponent_information_is_forbidden(self):
        forbidden = " ".join(pc.FORBIDDEN_SELECTOR_INPUTS)
        assert "opponent" in forbidden
        assert not any("opponent" in entry for entry in pc.ALLOWED_SELECTOR_INPUTS)

    def test_post_selection_path_is_the_accepted_phase7_one(self):
        assert pc.POST_SELECTION_PATH["reflection_probability"] == 0.5
        assert pc.POST_SELECTION_PATH["perturbation_probability"] == 0.5
        assert pc.POST_SELECTION_PATH["swap_counts"] == [1, 2, 3, 4, 5, 6]
        assert pc.POST_SELECTION_PATH["hamming_distance_window"] == [2, 12]


class TestBanksAndMatchups:
    def test_bank_sizes_and_family_balance(self):
        assert pc.VALIDATION_BANK_CASES == 128
        assert pc.VALIDATION_CASES_PER_FAMILY == 8
        assert pc.TEST_BANK_CASES == 512
        assert pc.TEST_CASES_PER_FAMILY == 32
        assert pc.VALIDATION_BANK_CASES == pc.VALIDATION_CASES_PER_FAMILY * len(FAMILY_IDS)
        assert pc.TEST_BANK_CASES == pc.TEST_CASES_PER_FAMILY * len(FAMILY_IDS)

    def test_six_matchups_with_the_frozen_neutral_arm_rule(self):
        assert len(pc.MATCHUPS) == 6
        by_token = {entry["token"]: entry for entry in pc.MATCHUPS}
        assert by_token[pc.MATCHUP_LEARNED_VS_NEUTRAL]["neutral_arm"] is False
        for token in (
            pc.MATCHUP_STRATEGIC,
            pc.MATCHUP_TACTICAL,
            pc.MATCHUP_PHASE8_ANCHOR,
            pc.MATCHUP_RANDOM,
            pc.MATCHUP_BASIC,
        ):
            assert by_token[token]["neutral_arm"] is True

    def test_move_behavior_is_greedy_float32_single_request_no_search(self):
        assert pc.EVAL_MOVE_BEHAVIOR == {
            "decision_mode": "greedy",
            "dtype": "float32",
            "batch_policy": "single_request",
            "search": "none",
        }


class TestSelectionAndGates:
    def test_selection_score_weights(self):
        assert pc.SELECTION_SCORE_WEIGHTS == {
            "delta_direct": 0.40,
            "delta_strategic": 0.30,
            "delta_tactical": 0.20,
            "delta_phase8_anchor": 0.10,
        }
        assert pc.SCORE_EXCLUDED_MATCHUPS == (pc.MATCHUP_RANDOM, pc.MATCHUP_BASIC)

    def test_validation_guards(self):
        assert pc.VALIDATION_RANDOM_MIN_EWR == 0.95
        assert pc.VALIDATION_BASIC_MIN_EWR == 0.80

    def test_eight_hard_gates_in_order(self):
        assert pc.HARD_GATE_IDS == ("A", "B", "C", "D", "E", "F", "G", "H")
        assert len(pc.HARD_GATES) == 8

    def test_frozen_gate_thresholds(self):
        assert pc.GATE_A["ordinary"]["ewr_min"] == 0.49
        assert pc.GATE_A["ordinary"]["lb_min"] == 0.47
        assert pc.GATE_A["improved"]["ewr_min"] == 0.52
        assert pc.GATE_A["improved"]["lb_min"] == 0.50
        assert pc.GATE_B["league_weights"] == {
            "delta_strategic": 0.45,
            "delta_tactical": 0.35,
            "delta_phase8_anchor": 0.20,
        }
        assert pc.GATE_B["delta_l_min"] == -0.01
        assert pc.GATE_B["lb_min"] == -0.03
        assert pc.GATE_C["lb_min"] == -0.03
        assert pc.GATE_D["random_overall_min"] == 0.95
        assert pc.GATE_D["random_red_min"] == 0.90
        assert pc.GATE_D["random_blue_min"] == 0.90
        assert pc.GATE_D["basic_min"] == 0.80
        assert pc.GATE_D["paired_lb_min"] == -0.03

    def test_diversity_thresholds(self):
        assert pc.DIVERSITY_THRESHOLDS == {
            "normalized_family_entropy_min": 0.85,
            "effective_families_min": 10.0,
            "family_probability_min": 0.015,
            "family_probability_max": 0.18,
            "within_family_normalized_base_entropy_min": 0.70,
            "max_conditional_base_probability": 0.10,
        }
        assert pc.SELECTOR_AUDIT_DRAWS == 100_000

    def test_four_classifications(self):
        assert set(pc.CLASSIFICATIONS) == {
            "PASS-IMPROVED",
            "PASS-NONINFERIOR",
            "FAIL",
            "BLOCKED",
        }

    def test_statistics_are_the_frozen_ones(self):
        assert pc.STATISTICS["replicates"] == 10_000
        assert pc.STATISTICS["confidence"] == 0.95
        assert pc.STATISTICS["method"] == "paired_unit_percentile_bootstrap"
        assert pc.STATISTICS["rng"] == "numpy_pcg64"
        assert pc.STATISTICS["validation_bootstrap_root"] == 2026081807
        assert pc.STATISTICS["final_bootstrap_root"] == 2026081808


class TestDocuments:
    def test_eight_documents_with_distinct_versions(self):
        documents = pc.contract_documents()
        assert len(documents) == 8
        assert set(documents) == set(pc.CONTRACT_VERSIONS)

    def test_documents_are_canonical_json_serializable(self):
        for name, document in pc.contract_documents().items():
            json.dumps(document, sort_keys=True, separators=(",", ":"))

    def test_digests_are_stable_across_rebuilds(self):
        assert pc.contract_digests() == pc.contract_digests()
        assert pc.contract_bundle_digest() == pc.contract_bundle_digest()

    def test_documents_carry_no_absolute_path(self):
        text = json.dumps(pc.contract_documents(), sort_keys=True)
        assert "/Volumes/" not in text
        assert "/Users/" not in text

    def test_system_document_leaves_the_unmeasured_slots_unbound(self):
        system = pc.system_document()
        slots = {entry["slot"] for entry in system["unbound_slots"]}
        assert slots == {
            "accepted_utility_model",
            "accepted_trait_scaler",
            "selected_selector_config",
        }
        assert system["bound_now"]["move_model"]["sha256"] == (
            pc.ACCEPTED_PHASE9_CHECKPOINT_SHA256
        )


class TestFrozenDigests:
    """The freeze itself. See the module docstring before changing a value."""

    EXPECTED = {
        "phase10_setup_contract_v1": None,
        "phase10_setup_outcome_corpus_v1": None,
        "phase10_setup_utility_v1": None,
        "phase10_setup_selector_v1": None,
        "phase10_selector_schedule_v1": None,
        "phase10_eval_bank_v1": None,
        "phase10_acceptance_v1": None,
        "phase10_system_v1": None,
    }

    def test_every_contract_has_a_pinned_digest(self):
        from tests.training.phase10_frozen_digests import CONTRACT_DIGESTS

        assert set(CONTRACT_DIGESTS) == set(self.EXPECTED)
        assert pc.contract_digests() == CONTRACT_DIGESTS

    def test_bundle_digest_is_pinned(self):
        from tests.training.phase10_frozen_digests import CONTRACT_BUNDLE_DIGEST

        assert pc.contract_bundle_digest() == CONTRACT_BUNDLE_DIGEST
