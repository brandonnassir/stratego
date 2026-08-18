"""Regression: the frozen Phase 10 paired evaluation banks and their isolation.

Both banks were built structurally by Agent 1 before any Phase 10 outcome
existed. These tests pin their identity, their family balance, the exact
fingerprint isolation the phase claims, and the order-independence that makes
a single case rebuildable on its own.

Nothing here plays a game or loads a model, which is what makes exercising
the sealed test bank legal: it is the `structural_audit` purpose the sealing
rules allow every agent.
"""

import pytest

from stratego.evaluation import phase10_banks as pb
from stratego.setups.families import FAMILY_IDS
from stratego.setups.sampler import load_library_index
from stratego.training import phase10_contract as pc
from stratego.training import phase10_seed as ps
from tests.training.phase10_frozen_digests import (
    BANK_DIGESTS,
    BANK_MANIFEST_DIGESTS,
    PHASE9_CANONICAL_HELD_OUT_IDENTITIES,
    PHASE9_HELD_OUT_SIDES,
    PHASE9_ISOLATION_SET_DIGEST,
    PHASE9_ISOLATION_SET_SIZE,
    PHASE9_RAW_HELD_OUT_BOARDS,
    PHASE9_TWO_ORIENTATION_CLASSES,
)


@pytest.fixture(scope="module")
def coverage():
    return pb.phase9_raw_board_coverage()


@pytest.fixture(scope="module")
def isolation():
    return pb.phase9_isolation_set()


@pytest.fixture(scope="module")
def banks(isolation):
    fingerprints, manifest = isolation
    return {
        name: pb.build_phase10_bank(name, fingerprints, manifest)
        for name in ("validation", "test")
    }


class TestIsolationSet:
    def test_the_phase9_set_is_the_pinned_one(self, isolation):
        fingerprints, manifest = isolation
        assert len(fingerprints) == PHASE9_ISOLATION_SET_SIZE
        assert manifest["set_digest"] == PHASE9_ISOLATION_SET_DIGEST

    def test_it_covers_both_accepted_phase9_banks(self, isolation):
        _, manifest = isolation
        versions = {source["bank_version"] for source in manifest["sources"]}
        assert versions == {"phase9_validation_bank_v1", "phase9_test_bank_v1"}


class TestStructure:
    @pytest.mark.parametrize(
        "name,count,per_family,split",
        [
            ("validation", 128, 8, "validation"),
            ("test", 512, 32, "test"),
        ],
    )
    def test_case_counts_and_family_balance(self, banks, name, count, per_family, split):
        cases, manifest = banks[name]
        assert len(cases) == count
        assert manifest["split"] == split
        counts = {family_id: 0 for family_id in FAMILY_IDS}
        for case in cases:
            counts[case.family_id] += 1
            assert case.split == split
        assert set(counts.values()) == {per_family}

    def test_case_ids_are_unique_across_both_banks(self, banks):
        identifiers = [
            case.case_id for cases, _ in banks.values() for case in cases
        ]
        assert len(set(identifiers)) == len(identifiers) == 640

    def test_every_case_fixes_exactly_three_arrangements(self, banks):
        for cases, _ in banks.values():
            for case in cases:
                assert len(case.frozen_fingerprints) == 3
                assert len(set(case.frozen_fingerprints)) == 3

    def test_selector_seeds_are_per_colour_and_arm_independent(self, banks):
        cases, _ = banks["validation"]
        case = cases[0]
        assert set(case.selector_seeds) == {"red", "blue"}
        assert case.selector_seeds["red"] != case.selector_seeds["blue"]
        for color in ("red", "blue"):
            assert case.selector_seeds[color] == ps.case_selector_seed(
                case.case_id, color, case.selector_seed_attempts[color]
            )

    def test_match_seeds_cover_every_frozen_matchup_and_both_games(self, banks):
        cases, _ = banks["validation"]
        case = cases[0]
        assert set(case.match_seeds) == set(pc.MATCHUP_TOKENS)
        for token, seeds in case.match_seeds.items():
            assert set(seeds) == {0, 1}
            assert seeds[0] != seeds[1]
            assert seeds[0] == ps.case_match_seed(case.case_id, 0, token)

    def test_the_opponent_setup_carries_the_case_family(self, banks):
        for cases, _ in banks.values():
            for case in cases[::37]:
                assert case.opponent_provenance["primary_family_id"] == case.family_id


class TestIsolationClaim:
    def test_no_frozen_arrangement_reuses_a_phase9_fingerprint(self, banks, isolation):
        fingerprints, _ = isolation
        for cases, _ in banks.values():
            for case in cases:
                for fingerprint in case.frozen_fingerprints:
                    assert fingerprint not in fingerprints

    def test_the_two_phase10_banks_do_not_share_a_fingerprint(self, banks):
        report = pb.cross_bank_isolation(banks["validation"][0], banks["test"][0])
        assert report["zero_overlap"]
        assert report["overlap_count"] == 0

    def test_the_rejection_walk_actually_fired(self, banks):
        _, manifest = banks["validation"]
        histogram = manifest["selector_seed_attempt_histogram"]
        assert sum(int(count) for count in histogram.values()) == 256
        assert any(int(attempt) > 0 for attempt in histogram)


class TestDeterminism:
    def test_a_case_rebuilds_from_identity_alone(self, banks, isolation):
        fingerprints, _ = isolation
        library = load_library_index()
        cases, _ = banks["validation"]
        for case in cases[::29]:
            assert pb.build_case("validation", case.case_index, fingerprints, index=library) == case

    def test_rebuilding_out_of_order_changes_nothing(self, banks, isolation):
        fingerprints, _ = isolation
        library = load_library_index()
        cases, _ = banks["validation"]
        for index in (100, 3, 64, 0):
            assert pb.build_case("validation", index, fingerprints, index=library) == cases[index]

    def test_bank_digests_are_pinned(self, banks):
        for name, (cases, manifest) in banks.items():
            assert pb.bank_digest(cases) == BANK_DIGESTS[name]
            assert manifest["bank_digest"] == BANK_DIGESTS[name]
            assert manifest["manifest_digest"] == BANK_MANIFEST_DIGESTS[name]

    def test_manifest_digest_excludes_run_measurements(self, banks):
        _, manifest = banks["validation"]
        mutated = dict(manifest)
        mutated["construction_run"] = {"duration_seconds": 999.0}
        assert pb.manifest_digest(mutated) == manifest["manifest_digest"]


class TestAudit:
    @pytest.mark.parametrize("name", ["validation", "test"])
    def test_the_full_structural_audit_passes(self, banks, isolation, name):
        fingerprints, _ = isolation
        cases, manifest = banks[name]
        audit = pb.audit_phase10_bank(
            name, cases, manifest, fingerprints, rebuild_sample_every=64
        )
        assert audit["all_pass"], {k: v for k, v in audit["checks"].items() if not v}
        assert audit["checks"]["phase9_fingerprint_overlap_zero"]
        assert audit["checks"]["split_isolation"]
        assert audit["checks"]["engine_valid"]
        assert audit["checks"]["provenance_rebuilds"]

    def test_a_tampered_digest_is_caught(self, banks, isolation):
        fingerprints, _ = isolation
        cases, manifest = banks["validation"]
        tampered = dict(manifest)
        tampered["bank_digest"] = "0" * 64
        audit = pb.audit_phase10_bank(
            "validation", cases, tampered, fingerprints, rebuild_sample_every=1024
        )
        assert not audit["checks"]["digest_matches_manifest"]
        assert not audit["all_pass"]

    def test_unknown_bank_is_refused(self):
        with pytest.raises(pb.Phase10BankError):
            pb.bank_specification("train")


class TestPhase9RawBoardCoverage:
    """The reconciliation: 1,280 sides, 1,233 raw boards, 1,184 identities.

    The accepted Phase 9 held-out universe can be counted as stored engine
    board strings or as canonical final-setup fingerprints, and the two
    counts differ. These tests prove the isolation set loses nothing by
    being stated canonically: every raw board maps into it, and every
    identity in it is reached.
    """

    def test_the_three_counts_are_the_accepted_ones(self, coverage):
        assert coverage["held_out_sides"] == PHASE9_HELD_OUT_SIDES
        assert coverage["distinct_raw_boards"] == PHASE9_RAW_HELD_OUT_BOARDS
        assert (
            coverage["distinct_canonical_identities"]
            == PHASE9_CANONICAL_HELD_OUT_IDENTITIES
        )
        assert coverage["isolation_set_size"] == PHASE9_CANONICAL_HELD_OUT_IDENTITIES

    def test_every_raw_board_maps_into_the_isolation_set(self, coverage):
        assert coverage["unmapped_raw_boards"] == []
        assert coverage["checks"]["every_raw_board_maps"]

    def test_the_mapping_is_surjective_onto_the_isolation_set(self, coverage):
        assert coverage["unreached_identities"] == []
        assert coverage["checks"]["mapping_is_surjective"]

    def test_each_raw_board_reproduces_its_recorded_fingerprint(self, coverage):
        assert coverage["round_trip_mismatches"] == []

    def test_the_duplicate_classes_explain_the_difference_exactly(self, coverage):
        assert coverage["duplicate_classes"] == PHASE9_TWO_ORIENTATION_CLASSES
        assert coverage["class_size_histogram"] == {
            "1": PHASE9_CANONICAL_HELD_OUT_IDENTITIES - PHASE9_TWO_ORIENTATION_CLASSES,
            "2": PHASE9_TWO_ORIENTATION_CLASSES,
        }
        assert (
            PHASE9_CANONICAL_HELD_OUT_IDENTITIES + PHASE9_TWO_ORIENTATION_CLASSES
            == PHASE9_RAW_HELD_OUT_BOARDS
        )

    def test_the_whole_coverage_audit_passes(self, coverage):
        assert coverage["all_pass"], {
            key: value for key, value in coverage["checks"].items() if not value
        }

    def test_the_audit_does_not_disturb_the_frozen_isolation_manifest(self, coverage):
        _, manifest = pb.phase9_isolation_set()
        assert coverage["isolation_set_digest"] == manifest["set_digest"]
        assert manifest["set_digest"] == PHASE9_ISOLATION_SET_DIGEST
