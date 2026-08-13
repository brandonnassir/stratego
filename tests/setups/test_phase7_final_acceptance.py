"""Phase 7 final acceptance: the freeze record must describe the live system.

These are the regressions that make the Agent 6 freeze durable rather than a
snapshot of one afternoon's run. They do not re-do Agent 6's expensive work —
no library regeneration, no 100,000-output corpus, no collection campaign —
they check that the frozen record and the live source still agree, so any later
change that silently drifts from `setup_library_v1` fails here.

**The freeze names live versions.** Every version string, digest, count and
sampler parameter in `agent_06_final_acceptance.json` is re-derived from the
live contracts, the live library file and the live sampler, not read back from
a sibling artifact.

**The frozen Phase 8 profile is the live default.** The exact probabilities,
intensity weights and Hamming window recorded in
`agent_06_sampler_profile.json` must still be what `sample_setup` would use.

**The regeneration claim stays falsifiable.** The recorded digests must still
be the digests of the materialized library, and a fresh isolated rebuild of a
deterministic sample must still reproduce its stored entries exactly.

**The perturbation identity stays corrected.** The production signature, the
`seed_encoding_v1` bijection and the rejection of tampered derived metadata are
pinned here, so Agent 4's authorized correction cannot regress unnoticed.
"""

import inspect
import json
from pathlib import Path

import pytest

from stratego.engine.constants import (
    IMPLEMENTATION_VERSION,
    OBSERVATION_CHANNELS,
    OBSERVATION_VERSION,
    RULES_VERSION,
)
from stratego.evaluation.setup_bank import SETUP_BANK_VERSION
from stratego.model.contract import MODEL_CONTRACT_VERSION
from stratego.setups import (
    BASE_SETUP_COUNT,
    BASES_PER_FAMILY,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_IDS,
    LIBRARY_JSONL_PATH,
    LIBRARY_MANIFEST_PATH,
    MAX_PERTURBATION_ATTEMPTS,
    MAX_SWAP_COUNT,
    MIN_SWAP_COUNT,
    PERTURBATION_VERSION,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    TEST_PER_FAMILY,
    TEST_TOTAL,
    TRAIN_PER_FAMILY,
    TRAIN_TOTAL,
    VALIDATION_PER_FAMILY,
    VALIDATION_TOTAL,
    decode_perturbation_seed,
    encode_perturbation_seed,
    entry_metadata_digest,
    library_content_digest,
    manifest_digest,
    perturb_setup,
    read_library_jsonl,
    read_manifest,
    rebuild_base_setup,
)
from stratego.setups.contracts import base_setup_id, isolated_rebuild_sample_indices
from stratego.setups.diversity import DIVERSITY_STANDARD_VERSION
from stratego.setups.identity import SetupLibraryError
from stratego.setups.sampler import (
    DEFAULT_PROFILE,
    PROFILES,
    SAMPLER_VERSION,
    load_library_index,
    rebuild_from_provenance,
    sample_setup,
)
from stratego.training.setup_source import (
    PROVENANCE_SCHEMA_VERSION,
    SETUP_SOURCE_VERSION,
    training_setup_source,
)
from stratego.training.trajectory import TRAJECTORY_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
LIBRARY_FILE = REPOSITORY_ROOT / LIBRARY_JSONL_PATH
MANIFEST_FILE = REPOSITORY_ROOT / LIBRARY_MANIFEST_PATH
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_7_data"

FINAL_ACCEPTANCE = DATA_DIRECTORY / "agent_06_final_acceptance.json"
LIBRARY_REGENERATION = DATA_DIRECTORY / "agent_06_library_regeneration.json"
SAMPLER_PROFILE = DATA_DIRECTORY / "agent_06_sampler_profile.json"

requires_production_library = pytest.mark.skipif(
    not LIBRARY_FILE.exists(),
    reason="production setup library not materialized yet",
)
requires_acceptance_artifacts = pytest.mark.skipif(
    not (
        FINAL_ACCEPTANCE.exists()
        and LIBRARY_REGENERATION.exists()
        and SAMPLER_PROFILE.exists()
    ),
    reason="Agent 6 acceptance artifacts not materialized yet",
)


@pytest.fixture(scope="module")
def acceptance() -> dict:
    return json.loads(FINAL_ACCEPTANCE.read_text())


@pytest.fixture(scope="module")
def regeneration() -> dict:
    return json.loads(LIBRARY_REGENERATION.read_text())


@pytest.fixture(scope="module")
def profile_record() -> dict:
    return json.loads(SAMPLER_PROFILE.read_text())


@pytest.fixture(scope="module")
def entries():
    return read_library_jsonl(LIBRARY_FILE)


@pytest.fixture(scope="module")
def index():
    return load_library_index(str(LIBRARY_FILE))


# ---------------------------------------------------------------------------
# The artifacts themselves
# ---------------------------------------------------------------------------


@requires_acceptance_artifacts
class TestAcceptanceArtifacts:
    def test_all_three_artifacts_declare_pass(
        self, acceptance, regeneration, profile_record
    ):
        assert acceptance["status"] == "PASS"
        assert regeneration["status"] == "PASS"
        assert profile_record["status"] == "PASS"

    def test_every_completion_gate_is_true(self, acceptance):
        """Every gate holds — except the one this test cannot observe.

        `full_repository_suite_green` is decided by the very suite run that
        contains this test, so the harness writes the artifacts once with that
        gate undecided, runs the suite against them, and re-emits with the
        totals. Asserting on it here would be circular: a first-pass artifact
        legitimately carries `null`, and a test that failed on that would make
        the suite fail in order to record that the suite failed. The suite
        result is evidenced by the run itself and recorded in `tests_after`.
        """
        gates = {
            name: value
            for name, value in acceptance["completion_gates"].items()
            if name != "full_repository_suite_green"
        }
        undecided = {name for name, value in gates.items() if value is None}
        assert not undecided, f"gates left undecided: {sorted(undecided)}"
        failing = [name for name, value in gates.items() if not value]
        assert not failing, f"failing gates: {failing}"

    def test_the_recorded_suite_result_is_green_when_it_is_recorded(self, acceptance):
        """Once a suite result has been recorded, it must be a green one."""
        tests_after = acceptance.get("tests_after")
        if tests_after is None:
            pytest.skip("artifacts written before the suite ran; re-emitted after")
        assert tests_after["failed"] == 0
        assert tests_after["returncode"] == 0
        assert tests_after["passed"] > acceptance["tests_before"]["passed"]

    def test_prerequisite_agents_are_recorded_as_pass(self, acceptance):
        statuses = acceptance["prerequisite_status"]
        assert set(statuses), "no prerequisite statuses recorded"
        assert all(value == "PASS" for value in statuses.values())

    def test_the_record_names_no_outcome_or_strength_evidence(self, acceptance):
        forbidden = ("win_rate", "elo", "policy_score", "game_outcome", "strength")
        serialized = json.dumps(acceptance).lower()
        for token in forbidden:
            assert f'"{token}"' not in serialized


# ---------------------------------------------------------------------------
# The frozen stack, re-derived from live source
# ---------------------------------------------------------------------------


@requires_acceptance_artifacts
class TestFrozenStack:
    def test_the_frozen_upstream_versions_are_the_live_ones(self, acceptance):
        frozen = acceptance["frozen_versions"]
        assert frozen["rules"] == RULES_VERSION == "stratego_project_v1"
        assert (
            frozen["reference_engine"]
            == IMPLEMENTATION_VERSION
            == "phase2_1_reference_1.2.0"
        )
        assert frozen["observation"] == OBSERVATION_VERSION == "observation_v2_1_127ch"
        assert frozen["observation_channels"] == OBSERVATION_CHANNELS == 127
        assert frozen["model_contract"] == MODEL_CONTRACT_VERSION == "model_contract_v2"
        assert frozen["trajectory"] == TRAJECTORY_VERSION == "trajectory_v1"
        assert frozen["phase_4_bank"] == SETUP_BANK_VERSION == "evaluation_setup_bank_v1"

    def test_the_frozen_phase_7_versions_are_the_live_ones(self, acceptance):
        frozen = acceptance["frozen_versions"]
        assert frozen["contract_version"] == SETUP_GENERATOR_CONTRACT_VERSION
        assert frozen["family_contract_version"] == SETUP_FAMILY_VERSION
        assert frozen["trait_schema_version"] == SETUP_TRAIT_VECTOR_VERSION
        assert frozen["library_version"] == SETUP_LIBRARY_VERSION
        assert frozen["perturbation_version"] == PERTURBATION_VERSION
        assert frozen["sampler_version"] == SAMPLER_VERSION
        assert frozen["setup_source_version"] == SETUP_SOURCE_VERSION
        assert frozen["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION
        assert frozen["master_seed"] == DEFAULT_LIBRARY_MASTER_SEED

    def test_the_diversity_standard_is_unchanged(self, acceptance):
        assert (
            acceptance["diversity_gate_summary"]["diversity_standard_version"]
            == DIVERSITY_STANDARD_VERSION
        )


# ---------------------------------------------------------------------------
# The library the freeze describes
# ---------------------------------------------------------------------------


@requires_production_library
@requires_acceptance_artifacts
class TestFrozenLibraryIdentity:
    def test_the_recorded_digests_are_the_live_digests(self, acceptance, entries):
        assert acceptance["library_digest"] == library_content_digest(entries)
        assert acceptance["manifest_digest"] == manifest_digest(
            read_manifest(MANIFEST_FILE)
        )
        assert acceptance["regeneration_digest"] == acceptance["library_digest"]
        assert acceptance["regeneration_mismatches"] == 0

    def test_the_regeneration_record_matches_the_live_library(
        self, regeneration, entries
    ):
        digests = regeneration["digests"]
        assert digests["regenerated_library_content_digest"] == library_content_digest(
            entries
        )
        assert digests["regenerated_entry_metadata_digest"] == entry_metadata_digest(
            entries
        )
        assert regeneration["checks"]["jsonl_bytes_identical"]
        assert regeneration["checks"]["isolated_rebuild_exact"]
        assert regeneration["checks"]["enumeration_order_independent"]

    def test_the_recorded_counts_are_the_live_counts(self, acceptance, entries):
        assert acceptance["setup_count"] == len(entries) == BASE_SETUP_COUNT
        assert set(acceptance["family_counts"]) == set(FAMILY_IDS)
        assert all(
            count == BASES_PER_FAMILY for count in acceptance["family_counts"].values()
        )
        assert acceptance["split_counts"] == {
            "train": TRAIN_TOTAL,
            "validation": VALIDATION_TOTAL,
            "test": TEST_TOTAL,
        }
        assert all(
            row
            == {
                "train": TRAIN_PER_FAMILY,
                "validation": VALIDATION_PER_FAMILY,
                "test": TEST_PER_FAMILY,
            }
            for row in acceptance["family_split_counts"].values()
        )

    def test_a_deterministic_sample_still_rebuilds_in_isolation(self, entries):
        """A base must still be reproducible without generating any other base."""
        stored = {entry.base_setup_id: entry for entry in entries}
        sample = [
            (family_id, base_index)
            for family_id in ("F00", "F07", "F15")
            for base_index in (0, 399, 400, 449, 450, 499)
        ]
        for family_id, base_index in sample:
            rebuilt = rebuild_base_setup(family_id, base_index)
            accepted = stored[base_setup_id(family_id, base_index)]
            assert rebuilt.canonical_setup == accepted.canonical_setup
            assert rebuilt.fingerprint == accepted.fingerprint
            assert rebuilt.split == accepted.split
            assert rebuilt.trait_vector == accepted.trait_vector

    def test_every_diversity_threshold_was_recorded_as_passing(self, acceptance):
        summary = acceptance["diversity_gate_summary"]
        assert summary["threshold_checks"] == summary["threshold_checks_passed"]
        assert summary["measurements_agreeing_with_agent_03"]
        assert summary["threshold_checks"] > 0


# ---------------------------------------------------------------------------
# The frozen Phase 8 sampling profile
# ---------------------------------------------------------------------------


@requires_production_library
@requires_acceptance_artifacts
class TestFrozenSamplerProfile:
    def test_the_frozen_profile_is_an_accepted_agent_4_profile(self, acceptance):
        decision = acceptance["default_phase_8_sampler_profile"]
        assert decision["profile_name"] in PROFILES

    def test_the_frozen_profile_is_the_live_default(self, acceptance):
        decision = acceptance["default_phase_8_sampler_profile"]
        live = PROFILES[decision["profile_name"]]
        assert live is DEFAULT_PROFILE
        assert decision["perturbation_probability"] == live.perturbation_probability
        assert decision["reflection_probability"] == live.reflection_probability
        assert tuple(decision["intensity_weights"]) == live.intensity_weights
        assert tuple(decision["swap_counts"]) == live.swap_counts

    def test_the_frozen_profile_keeps_the_common_contract_invariants(self, acceptance):
        decision = acceptance["default_phase_8_sampler_profile"]
        assert decision["split"] == "train"
        assert decision["family_selection"].startswith("uniform")
        assert decision["base_selection"].startswith("uniform")
        assert decision["reflection_probability"] == 0.5

    def test_the_frozen_window_and_budget_are_the_live_constants(self, acceptance):
        decision = acceptance["default_phase_8_sampler_profile"]
        assert decision["hamming_window"] == [2 * MIN_SWAP_COUNT, 2 * MAX_SWAP_COUNT]
        assert decision["perturbation_max_attempts"] == MAX_PERTURBATION_ATTEMPTS

    def test_the_decision_used_only_structural_evidence(self, profile_record):
        admissibility = profile_record["evidence_admissibility"]
        assert "game outcomes" in admissibility["prohibited_and_unused"]
        assert "win rate" in admissibility["prohibited_and_unused"]
        assert "family balance" in admissibility["permitted_and_used"]

    def test_the_eliminated_profiles_were_eliminated_structurally(self, profile_record):
        """The two rejections must rest on measurements, not on preference."""
        evidence = profile_record["evidence"]
        # Reflection is class-invariant, so this is a structural ceiling that no
        # number of draws can lift.
        assert (
            evidence["reflection_only_v1"]["effective_support"]["distinct_novel_classes"]
            == 0
        )
        assert evidence["reflection_only_v1"]["effective_support"][
            "class_support_bounded_by_library"
        ]
        # A profile that perturbs every draw shows the learner no curated base.
        assert evidence["perturbation_only_v1"]["library_faithful_fraction"] == 0.0
        assert evidence["neutral_v1"]["library_faithful_fraction"] > 0.0
        assert (
            evidence["neutral_v1"]["effective_support"]["distinct_novel_classes"] > 0
        )

    def test_the_production_training_source_uses_the_frozen_profile(self, acceptance):
        decision = acceptance["default_phase_8_sampler_profile"]
        source = training_setup_source()
        assert source.split == "train"
        assert source.profile == decision["profile_name"]

    def test_the_frozen_profile_still_samples_deterministically(self, acceptance, index):
        decision = acceptance["default_phase_8_sampler_profile"]
        for seed in (0, 17, 2_026):
            first = sample_setup("train", seed, profile=decision["profile_name"], index=index)
            second = sample_setup("train", seed, profile=decision["profile_name"], index=index)
            assert first.canonical == second.canonical
            assert first.provenance == second.provenance
            assert first.split == "train"


# ---------------------------------------------------------------------------
# The corrected perturbation identity
# ---------------------------------------------------------------------------


@requires_production_library
@requires_acceptance_artifacts
class TestPerturbationIdentityStaysCorrected:
    def test_the_production_signature_carries_the_identity_and_nothing_else(self):
        assert list(inspect.signature(perturb_setup).parameters) == [
            "base_canonical",
            "family_id",
            "perturbation_seed",
        ]

    def test_the_seed_encoding_is_bijective_over_the_frozen_window(self):
        for swap_count in range(MIN_SWAP_COUNT, MAX_SWAP_COUNT + 1):
            for raw_seed in (0, 1, 2**31 - 1, 2**52 + 9):
                seed = encode_perturbation_seed(swap_count, raw_seed)
                assert decode_perturbation_seed(seed) == (swap_count, raw_seed)

    @pytest.mark.parametrize("low_bits", [6, 7])
    def test_an_invalid_encoding_is_rejected(self, low_bits):
        with pytest.raises(SetupLibraryError):
            decode_perturbation_seed((99 << 3) | low_bits)

    def test_the_same_identity_yields_the_same_descendant(self, index):
        entry = index.base(base_setup_id("F05", 42))
        seed = encode_perturbation_seed(4, 777_777)
        first = perturb_setup(entry.canonical_setup, entry.family_id, seed)
        second = perturb_setup(entry.canonical_setup, entry.family_id, seed)
        assert first.canonical == second.canonical
        assert first.swap_count == 4

    def test_tampered_derived_metadata_is_rejected_by_rebuild(self, index):
        for seed in range(64):
            sampled = sample_setup("train", seed, index=index)
            if not sampled.provenance["perturbation_applied"]:
                continue
            corrupted = dict(sampled.provenance)
            corrupted["perturbation_max_attempts"] = MAX_PERTURBATION_ATTEMPTS - 1
            with pytest.raises(SetupLibraryError):
                rebuild_from_provenance(corrupted, index=index)
            corrupted = dict(sampled.provenance)
            corrupted["perturbation_swap_count"] = 1 + (
                int(corrupted["perturbation_swap_count"]) % MAX_SWAP_COUNT
            )
            with pytest.raises(SetupLibraryError):
                rebuild_from_provenance(corrupted, index=index)
            return
        pytest.fail("no perturbed sample found in the first 64 draws")


# ---------------------------------------------------------------------------
# Pipeline and isolation claims the freeze rests on
# ---------------------------------------------------------------------------


@requires_acceptance_artifacts
class TestIntegrationClaims:
    def test_the_accepted_campaign_replay_is_recorded_as_exact(self, acceptance):
        replay = acceptance["pipeline_integration_summary"]["accepted_campaign_replay"]
        assert replay["mismatches"] == 0
        assert replay["rows_compared"] >= 8_000

    def test_the_live_spot_check_reproduced_the_accepted_campaign(self, acceptance):
        live = acceptance["pipeline_integration_summary"]["live_spot_check"]
        assert live["executed"]
        cross = live["accepted_campaign_cross_check"]
        assert cross["overlapping_logical_games"] > 0
        assert cross["field_mismatches"] == 0
        assert live["model_weights_mutated"] is False
        assert live["optimizer_steps"] == 0

    def test_the_phase_4_evaluation_bank_is_unchanged(self, acceptance):
        bank = acceptance["pipeline_integration_summary"]["phase_4_evaluation_bank"]
        assert bank["unchanged"]
        assert bank["before"]["digest"] == bank["after"]["digest"]
        assert bank["after"]["bank_version"] == SETUP_BANK_VERSION
        assert bank["after"]["validation_failure_count"] == 0

    def test_validation_and_test_access_stay_behind_an_explicit_request(
        self, acceptance
    ):
        access = acceptance["pipeline_integration_summary"]["split_access"]
        for split in ("validation", "test"):
            assert access[split]["purpose"] == "evaluation_audit"
            assert access[split]["justification_recorded"]
            assert access[split]["outside_train_range"]

    def test_the_observer_safe_boundary_is_unchanged(self, acceptance):
        safety = acceptance["observer_safety_summary"]
        assert safety["observation_version"] == OBSERVATION_VERSION
        assert safety["observation_channels"] == OBSERVATION_CHANNELS == 127
        assert safety["trajectory_version"] == TRAJECTORY_VERSION

    def test_the_procedural_stress_summary_records_no_hard_failure(self, acceptance):
        summary = acceptance["procedural_stress_summary"]
        assert summary["outputs"] >= 100_000
        assert all(value == 0 for value in summary["hard_requirements"].values())
        assert all(
            row["match"] for row in summary["agent_04_reproduction"].values()
        )
        assert all(summary["identity_semantics"]["checks"].values())
        assert summary["support_expansion"]["distinct_classes_from_100k_outputs"] > (
            4 * BASE_SETUP_COUNT
        )
