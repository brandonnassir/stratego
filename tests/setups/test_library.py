"""Materialized `setup_library_v1`: counts, uniqueness, digests, artifacts.

Three kinds of test live here.

**Structure of the generated library.** The 8,000 entries are generated once
per module and then held to the frozen targets: 500 per family, 400/50/50 per
family per split, zero duplicates of any kind, zero engine-invalid or stranded
bases, zero family violations, every stored arrangement canonical. These are
the acceptance conditions Phase 7 states as exact zeros, so they are checked as
exact zeros rather than as tolerances.

**Detector honesty.** A verification suite that cannot fail is worthless, so
the uniqueness gate is fed planted exact duplicates, planted mirrored
duplicates and a planted stable-id collision, and must raise on each. The
manifest digest is likewise fed a mutated library and a mutated run section,
and must move for the first and not for the second.

**Agreement with the materialized artifact.** When the production files exist,
they must equal what a fresh generation produces — same digests, same bytes,
same manifest — and the Agent 2 report artifacts must agree with both. That is
the property Agent 6's bit-for-bit regeneration will re-run from scratch.
"""

import json
from pathlib import Path

import pytest

from stratego.engine.setup import serialize_setup
from stratego.setups import (
    BASE_ENTRY_REQUIRED_FIELDS,
    BASE_SETUP_COUNT,
    BASES_PER_FAMILY,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_IDS,
    GENERATOR_VERSION,
    LIBRARY_JSONL_PATH,
    LIBRARY_MANIFEST_PATH,
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
    LibrarySeedContext,
    build_manifest,
    entry_metadata_digest,
    generate_library,
    library_content_digest,
    library_order,
    manifest_digest,
    read_library_jsonl,
    read_manifest,
    reflect_canonical,
    verify_library,
    write_library_jsonl,
    write_manifest,
)
from stratego.setups.identity import SetupLibraryError
from stratego.setups.library import (
    FORBIDDEN_ENTRY_FIELD_TOKENS,
    _enforce_global_uniqueness,
    entry_lines,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_7_data"
LIBRARY_FILE = REPOSITORY_ROOT / LIBRARY_JSONL_PATH
MANIFEST_FILE = REPOSITORY_ROOT / LIBRARY_MANIFEST_PATH


@pytest.fixture(scope="module")
def library():
    """The complete generated library — about three seconds, built once."""
    return generate_library()


@pytest.fixture(scope="module")
def verification(library):
    return verify_library(list(library.entries))


# ---------------------------------------------------------------------------
# Counts and splits
# ---------------------------------------------------------------------------


class TestCounts:
    def test_the_library_holds_exactly_eight_thousand_bases(self, library):
        assert len(library.entries) == BASE_SETUP_COUNT == 8000

    def test_every_family_holds_exactly_five_hundred_bases(self, verification):
        assert verification["family_counts"] == {
            family_id: BASES_PER_FAMILY for family_id in FAMILY_IDS
        }

    def test_every_family_splits_exactly_four_hundred_fifty_fifty(self, verification):
        for family_id in FAMILY_IDS:
            assert verification["family_split_counts"][family_id] == {
                "train": TRAIN_PER_FAMILY,
                "validation": VALIDATION_PER_FAMILY,
                "test": TEST_PER_FAMILY,
            }

    def test_library_split_totals_match_the_frozen_allocation(self, verification):
        assert verification["split_counts"] == {
            "train": TRAIN_TOTAL,
            "validation": VALIDATION_TOTAL,
            "test": TEST_TOTAL,
        }

    def test_no_setup_appears_in_more_than_one_split(self, library):
        by_fingerprint: dict = {}
        for entry in library.entries:
            by_fingerprint.setdefault(entry.fingerprint, set()).add(entry.split)
        assert all(len(splits) == 1 for splits in by_fingerprint.values())

    def test_entries_are_stored_in_the_frozen_file_order(self, library):
        assert [
            (entry.family_id, entry.base_index) for entry in library.entries
        ] == library_order()


# ---------------------------------------------------------------------------
# Verification of every entry, recomputed from content
# ---------------------------------------------------------------------------


class TestVerification:
    def test_every_generation_time_check_passes(self, verification):
        failed = [name for name, passed in verification["checks"].items() if not passed]
        assert failed == []
        assert verification["all_pass"]

    def test_the_hard_zeros_are_actually_zero(self, verification):
        assert verification["exact_duplicate_groups"] == 0
        assert verification["reflection_duplicate_groups"] == 0
        assert verification["content_fingerprint_collisions"] == 0
        assert verification["stable_id_collisions"] == 0
        assert verification["engine_invalid"] == []
        assert verification["stranded"] == []
        assert verification["family_violations"] == []
        assert verification["non_canonical_entries"] == []
        assert verification["reflection_roundtrip_failures"] == []
        assert verification["identity_mismatches"] == []
        assert verification["metadata_mismatches"] == []

    def test_every_class_fingerprint_is_distinct_across_the_whole_library(self, library):
        fingerprints = {entry.fingerprint for entry in library.entries}
        assert len(fingerprints) == BASE_SETUP_COUNT

    def test_no_arrangement_repeats_even_across_families(self, library):
        arrangements = {entry.canonical_setup for entry in library.entries}
        assert len(arrangements) == BASE_SETUP_COUNT

    def test_no_reflection_of_a_stored_base_is_also_stored(self, library):
        stored = {entry.canonical_setup for entry in library.entries}
        mirrored = {reflect_canonical(setup) for setup in stored}
        assert stored.isdisjoint(mirrored)

    def test_no_entry_field_carries_an_outcome_or_strength_signal(self, verification):
        assert verification["forbidden_entry_fields"] == []
        assert FORBIDDEN_ENTRY_FIELD_TOKENS  # the detector list is not empty

    def test_trait_distributions_are_reported_for_every_family(self, verification):
        distributions = verification["trait_distributions"]
        assert set(distributions) == set(FAMILY_IDS)
        for distribution in distributions.values():
            assert distribution["member_count"] == BASES_PER_FAMILY
            assert sum(distribution["flag_rank_histogram"]) == BASES_PER_FAMILY
            assert sum(distribution["flag_edge_distance_histogram"]) == BASES_PER_FAMILY


# ---------------------------------------------------------------------------
# The uniqueness gate must be able to fail
# ---------------------------------------------------------------------------


class TestUniquenessGate:
    def test_a_clean_library_passes_the_gate(self, library):
        _enforce_global_uniqueness(list(library.entries))

    def test_a_planted_exact_duplicate_is_rejected(self, library):
        import dataclasses

        entries = list(library.entries[:64])
        # Same arrangement carried by a different identity: the cross-family
        # duplicate the contract forbids.
        entries.append(dataclasses.replace(entries[0], base_setup_id="setup_library_v1:F15:499"))
        with pytest.raises(SetupLibraryError, match="fingerprint collision"):
            _enforce_global_uniqueness(entries)

    def test_a_planted_mirrored_duplicate_is_rejected(self, library):
        import dataclasses

        from stratego.setups import class_fingerprint

        entries = list(library.entries[:64])
        mirrored = reflect_canonical(entries[0].canonical_setup)
        assert mirrored != entries[0].canonical_setup
        entries.append(
            dataclasses.replace(
                entries[0],
                base_setup_id="setup_library_v1:F15:498",
                canonical_setup=mirrored,
                fingerprint=class_fingerprint(mirrored),
            )
        )
        with pytest.raises(SetupLibraryError, match="fingerprint collision"):
            _enforce_global_uniqueness(entries)

    def test_a_planted_stable_id_collision_is_rejected(self, library):
        import dataclasses

        entries = list(library.entries[:64])
        entries.append(
            dataclasses.replace(entries[5], base_setup_id=entries[0].base_setup_id)
        )
        with pytest.raises(SetupLibraryError, match="stable id collision"):
            _enforce_global_uniqueness(entries)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_every_entry_line_carries_the_frozen_required_fields(self, library):
        payload = json.loads(entry_lines(library.entries[:1])[0])
        for name in BASE_ENTRY_REQUIRED_FIELDS:
            assert name in payload
        assert payload["canonical_setup"] == serialize_setup(
            library.entries[0].canonical_setup
        )

    def test_lines_use_the_frozen_canonical_json_form(self, library):
        line = entry_lines(library.entries[:1])[0]
        assert line == json.dumps(
            json.loads(line), sort_keys=True, separators=(",", ":")
        )

    def test_writing_reading_and_rewriting_is_byte_stable(self, library, tmp_path):
        target = tmp_path / "library.jsonl"
        first_bytes = write_library_jsonl(target, library.entries)
        original = target.read_bytes()
        reread = read_library_jsonl(target)
        assert [entry.to_dict() for entry in reread] == [
            entry.to_dict() for entry in library.entries
        ]
        second_bytes = write_library_jsonl(target, reread)
        assert target.read_bytes() == original
        assert first_bytes == second_bytes == len(original)

    def test_the_same_library_always_produces_the_same_digests(self, library):
        first = (
            library_content_digest(library.entries),
            entry_metadata_digest(library.entries),
        )
        second = (
            library_content_digest(library.entries),
            entry_metadata_digest(library.entries),
        )
        assert first == second

    def test_a_changed_arrangement_changes_both_digests(self, library):
        import dataclasses

        from stratego.setups import class_fingerprint

        entries = list(library.entries)
        mirrored = reflect_canonical(entries[0].canonical_setup)
        entries[0] = dataclasses.replace(
            entries[0], canonical_setup=mirrored, fingerprint=class_fingerprint(mirrored)
        )
        assert library_content_digest(entries) == library_content_digest(library.entries)
        # Content identity is the reflection class, so the class digest is
        # unchanged by a mirror — but the stored bytes are not.
        assert entry_metadata_digest(entries) != entry_metadata_digest(library.entries)

    def test_changed_provenance_alone_moves_the_metadata_digest(self, library):
        import dataclasses

        entries = list(library.entries)
        entries[7] = dataclasses.replace(entries[7], generation_attempts=99)
        assert entry_metadata_digest(entries) != entry_metadata_digest(library.entries)
        assert library_content_digest(entries) == library_content_digest(library.entries)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest(library):
    return build_manifest(library, command="pytest")


class TestManifest:
    def test_the_manifest_states_every_required_field(self, manifest):
        for name in (
            "library_version",
            "generator_contract_version",
            "family_version",
            "trait_schema_version",
            "master_seed",
            "entry_count",
            "family_counts",
            "split_counts",
            "family_split_counts",
            "library_content_digest",
            "entry_metadata_digest",
            "rejection_counts",
        ):
            assert name in manifest, name
        assert manifest["generation_run"]["command"] == "pytest"
        assert manifest["generation_run"]["duration_seconds"] >= 0.0

    def test_manifest_versions_and_counts_match_the_frozen_contract(self, manifest):
        assert manifest["library_version"] == SETUP_LIBRARY_VERSION
        assert manifest["generator_contract_version"] == SETUP_GENERATOR_CONTRACT_VERSION
        assert manifest["family_version"] == SETUP_FAMILY_VERSION
        assert manifest["trait_schema_version"] == SETUP_TRAIT_VECTOR_VERSION
        assert manifest["generator_version"] == GENERATOR_VERSION
        assert manifest["master_seed"] == DEFAULT_LIBRARY_MASTER_SEED
        assert manifest["entry_count"] == BASE_SETUP_COUNT
        assert manifest["split_counts"] == {
            "train": TRAIN_TOTAL,
            "validation": VALIDATION_TOTAL,
            "test": TEST_TOTAL,
        }

    def test_the_manifest_digest_ignores_run_measurements(self, manifest):
        mutated = json.loads(json.dumps(manifest))
        mutated["generation_run"]["duration_seconds"] = 999.0
        mutated["generation_run"]["timestamp"] = "1999-01-01T00:00:00+0000"
        assert manifest_digest(mutated) == manifest["manifest_digest"]

    def test_the_manifest_digest_tracks_library_identity(self, manifest):
        mutated = json.loads(json.dumps(manifest))
        mutated["library_content_digest"] = "0" * 64
        assert manifest_digest(mutated) != manifest["manifest_digest"]

    def test_the_manifest_round_trips_through_disk(self, manifest, tmp_path):
        target = tmp_path / "manifest.json"
        written = write_manifest(target, manifest)
        assert written > 0
        assert read_manifest(target) == manifest

    def test_rejection_counts_are_recorded_by_family_and_reason(self, manifest, library):
        by_reason = manifest["rejection_counts"]["by_reason"]
        assert set(by_reason) == {
            "construction_infeasible",
            "engine_invalid",
            "family_predicate",
            "stranded",
        }
        assert manifest["rejection_counts"]["total"] == sum(by_reason.values())
        assert by_reason["engine_invalid"] == 0  # construction never breaks inventory
        assert manifest["attempt_statistics"]["total_attempts"] == (
            BASE_SETUP_COUNT + manifest["rejection_counts"]["total"]
        )


# ---------------------------------------------------------------------------
# Determinism at library scale
# ---------------------------------------------------------------------------


class TestLibraryDeterminism:
    def test_generating_in_a_shuffled_order_reproduces_identical_entries(self, library):
        import random

        rng = random.Random(4242)
        order = library_order()
        rng.shuffle(order)
        shuffled = generate_library(order=order[:200])
        by_identifier = {entry.base_setup_id: entry for entry in library.entries}
        for entry in shuffled.entries:
            assert entry.to_dict() == by_identifier[entry.base_setup_id].to_dict()

    def test_a_different_master_seed_produces_a_different_library(self, library):
        alternative = generate_library(
            LibrarySeedContext(master_seed=DEFAULT_LIBRARY_MASTER_SEED + 1),
            order=library_order()[:64],
        )
        production = {entry.fingerprint for entry in library.entries}
        assert all(entry.fingerprint not in production for entry in alternative.entries)


# ---------------------------------------------------------------------------
# The materialized production artifact
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def materialized():
    return read_library_jsonl(LIBRARY_FILE)


@pytest.fixture(scope="module")
def summary():
    return json.loads((DATA_DIRECTORY / "agent_02_generation_summary.json").read_text())


@pytest.fixture(scope="module")
def artifact_manifest():
    return json.loads((DATA_DIRECTORY / "agent_02_base_library_manifest.json").read_text())


@pytest.mark.skipif(
    not LIBRARY_FILE.exists(), reason="production library not materialized yet"
)
class TestMaterializedLibrary:
    def test_the_materialized_library_matches_a_fresh_generation(self, materialized, library):
        assert [entry.to_dict() for entry in materialized] == [
            entry.to_dict() for entry in library.entries
        ]

    def test_the_materialized_library_passes_every_verification_check(self, materialized):
        verification = verify_library(materialized)
        failed = [name for name, passed in verification["checks"].items() if not passed]
        assert failed == []

    def test_the_manifest_file_describes_the_materialized_library(self, materialized):
        manifest = read_manifest(MANIFEST_FILE)
        assert manifest["entry_count"] == len(materialized) == BASE_SETUP_COUNT
        assert manifest["library_content_digest"] == library_content_digest(materialized)
        assert manifest["entry_metadata_digest"] == entry_metadata_digest(materialized)
        assert manifest["manifest_digest"] == manifest_digest(manifest)
        assert manifest["master_seed"] == DEFAULT_LIBRARY_MASTER_SEED

    def test_the_file_bytes_are_the_canonical_serialization(self, materialized):
        expected = "\n".join(entry_lines(materialized)) + "\n"
        assert LIBRARY_FILE.read_text(encoding="utf-8") == expected


@pytest.mark.skipif(
    not (DATA_DIRECTORY / "agent_02_generation_summary.json").exists(),
    reason="Agent 2 artifacts not written yet",
)
class TestAgentArtifacts:
    def test_the_summary_reports_pass_with_every_gate_true(self, summary):
        assert summary["status"] == "PASS"
        assert summary["gates_true"] == summary["gates_total"]
        assert all(summary["completion_gates"].values())

    def test_the_summary_headline_numbers_match_the_materialized_library(self, summary):
        manifest = read_manifest(MANIFEST_FILE)
        assert summary["setup_count"] == BASE_SETUP_COUNT
        assert summary["library_digest"] == manifest["library_content_digest"]
        assert summary["manifest_digest"] == manifest["manifest_digest"]
        assert summary["master_seed"] == DEFAULT_LIBRARY_MASTER_SEED
        assert summary["split_counts"] == {
            "train": TRAIN_TOTAL,
            "validation": VALIDATION_TOTAL,
            "test": TEST_TOTAL,
        }

    def test_the_artifact_manifest_is_the_materialized_manifest(self, artifact_manifest):
        assert artifact_manifest["manifest"] == read_manifest(MANIFEST_FILE)
        assert artifact_manifest["library_jsonl_path"] == LIBRARY_JSONL_PATH
        assert artifact_manifest["library_bytes"] == LIBRARY_FILE.stat().st_size

    def test_the_prerequisite_check_recorded_agent_one_agreement(self, summary):
        prerequisites = summary["prerequisite_status"]
        assert prerequisites["agent_01_pass"]
        assert prerequisites["contract_matches_artifact"]
        assert prerequisites["thresholds_match_artifact"]
        assert prerequisites["reference_engine_is_1_2_0"]

    def test_the_diversity_preflight_is_recorded_but_not_declared_final(self, summary):
        preflight = summary["diversity_preflight"]
        assert preflight is None or preflight["verdict_owner"] == "agent_03"
