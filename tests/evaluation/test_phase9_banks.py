"""Regression: the frozen Phase 9 evaluation banks stay exactly what Agent 1 froze.

The two bank digests pinned here are frozen identity: the validation bank
selects every Phase 9 checkpoint and the test bank is the sealed final
evaluation, so a changed digest means a different phase. Construction is
deterministic from frozen constants, which is what lets this suite rebuild
the banks and require byte-identical results.
"""

import pytest

from stratego.engine.constants import BLUE, RED
from stratego.evaluation.match_spec import (
    build_paired_schedule,
    schedule_matches,
    validate_schedule,
)
from stratego.evaluation.phase9_banks import (
    Phase9BankError,
    audit_phase9_bank,
    bank_specification,
    build_case,
    build_phase9_bank,
    case_family,
    resolve_case_side,
)
from stratego.evaluation.registry import policy_ref
from stratego.evaluation.setup_bank import bank_digest
from stratego.setups.contracts import parse_base_setup_id, split_for_base_index
from stratego.setups.families import FAMILY_IDS

#: The frozen bank digests. These are identity, not implementation detail.
FROZEN_VALIDATION_BANK_DIGEST = (
    "3d28d544f6669129b12c13e4e3738aa36d1a99e4af8f6685bbb032793701ee4a"
)
FROZEN_TEST_BANK_DIGEST = (
    "f38e405559fc7c04b0832b1d3a4e3d82cd68ffff29bc1a9af456a3940e1de6a7"
)


@pytest.fixture(scope="module")
def validation_bank():
    return build_phase9_bank("validation")


@pytest.fixture(scope="module")
def test_bank():
    return build_phase9_bank("test")


class TestSpecifications:
    def test_the_two_banks(self):
        validation = bank_specification("validation")
        assert validation["bank_version"] == "phase9_validation_bank_v1"
        assert validation["split"] == "validation"
        assert validation["cases_per_family"] == 8
        assert validation["case_count"] == 128
        test = bank_specification("test")
        assert test["bank_version"] == "phase9_test_bank_v1"
        assert test["split"] == "test"
        assert test["cases_per_family"] == 32
        assert test["case_count"] == 512

    def test_unknown_bank_is_refused(self):
        with pytest.raises(Phase9BankError):
            bank_specification("train")


class TestCaseFamilyArithmetic:
    def test_family_major_layout(self):
        assert case_family(0, 8) == "F00"
        assert case_family(7, 8) == "F00"
        assert case_family(8, 8) == "F01"
        assert case_family(127, 8) == "F15"
        assert case_family(0, 32) == "F00"
        assert case_family(31, 32) == "F00"
        assert case_family(32, 32) == "F01"
        assert case_family(511, 32) == "F15"

    def test_out_of_range_is_refused(self):
        with pytest.raises(Phase9BankError):
            case_family(128, 8)
        with pytest.raises(Phase9BankError):
            case_family(512, 32)


class TestDeterministicConstruction:
    def test_case_side_matches_the_requested_family(self):
        sampled, attempt, seed = resolve_case_side(
            "phase9_validation_bank_v1", "validation", "F03", 2, "red"
        )
        assert sampled.family_id == "F03"
        assert sampled.split == "validation"
        assert attempt >= 0
        again, attempt_again, seed_again = resolve_case_side(
            "phase9_validation_bank_v1", "validation", "F03", 2, "red"
        )
        assert (again.canonical, attempt_again, seed_again) == (
            sampled.canonical, attempt, seed,
        )

    def test_bad_side_is_refused(self):
        with pytest.raises(Phase9BankError):
            resolve_case_side("phase9_validation_bank_v1", "validation", "F03", 2, "green")

    def test_build_case_is_deterministic(self):
        first_pair, first_provenance = build_case("validation", 17)
        second_pair, second_provenance = build_case("validation", 17)
        assert first_pair == second_pair
        assert first_provenance == second_provenance

    def test_sides_are_independent_draws(self):
        pair, provenance = build_case("validation", 5)
        assert provenance["red"]["accepted_draw_seed"] != (
            provenance["blue"]["accepted_draw_seed"]
        )


class TestValidationBank:
    def test_frozen_digest(self, validation_bank):
        bank, manifest = validation_bank
        assert bank_digest(bank) == FROZEN_VALIDATION_BANK_DIGEST
        assert manifest["bank_digest"] == FROZEN_VALIDATION_BANK_DIGEST

    def test_shape_and_family_balance(self, validation_bank):
        bank, manifest = validation_bank
        assert len(bank.pairs) == 128
        assert bank.bank_version == "phase9_validation_bank_v1"
        counts = {family_id: 0 for family_id in FAMILY_IDS}
        for record in manifest["case_provenance"]:
            counts[record["family_id"]] += 1
        assert all(count == 8 for count in counts.values())

    def test_family_purity_and_split_isolation(self, validation_bank):
        _bank, manifest = validation_bank
        for record in manifest["case_provenance"]:
            for side in ("red", "blue"):
                assert record[side]["primary_family_id"] == record["family_id"]
                assert record[side]["split"] == "validation"
                _, _, base_index = parse_base_setup_id(record[side]["base_setup_id"])
                assert split_for_base_index(base_index) == "validation"

    def test_structural_audit_passes(self, validation_bank):
        bank, manifest = validation_bank
        audit = audit_phase9_bank("validation", bank, manifest, rebuild_sample_every=32)
        assert audit["all_pass"], audit["checks"]


class TestTestBank:
    def test_frozen_digest(self, test_bank):
        bank, manifest = test_bank
        assert bank_digest(bank) == FROZEN_TEST_BANK_DIGEST
        assert manifest["bank_digest"] == FROZEN_TEST_BANK_DIGEST

    def test_shape_and_family_balance(self, test_bank):
        bank, manifest = test_bank
        assert len(bank.pairs) == 512
        assert bank.bank_version == "phase9_test_bank_v1"
        counts = {family_id: 0 for family_id in FAMILY_IDS}
        for record in manifest["case_provenance"]:
            counts[record["family_id"]] += 1
        assert all(count == 32 for count in counts.values())

    def test_split_isolation(self, test_bank):
        _bank, manifest = test_bank
        for record in manifest["case_provenance"]:
            for side in ("red", "blue"):
                assert record[side]["split"] == "test"
                _, _, base_index = parse_base_setup_id(record[side]["base_setup_id"])
                assert split_for_base_index(base_index) == "test"

    def test_structural_audit_passes(self, test_bank):
        bank, manifest = test_bank
        audit = audit_phase9_bank("test", bank, manifest, rebuild_sample_every=64)
        assert audit["all_pass"], audit["checks"]

    def test_disjoint_from_the_validation_bank(self, validation_bank, test_bank):
        validation_positions = {
            (pair.red_setup, pair.blue_setup) for pair in validation_bank[0].pairs
        }
        test_positions = {
            (pair.red_setup, pair.blue_setup) for pair in test_bank[0].pairs
        }
        assert not validation_positions & test_positions


class TestColorPairingExactness:
    def test_paired_units_swap_colors_on_the_same_board(self, validation_bank):
        bank, _manifest = validation_bank
        candidate = policy_ref("strategic_rule_based")
        opponent = policy_ref("tactical_rule_based")
        units = build_paired_schedule(
            candidate,
            opponent,
            range(4),
            setup_bank_version=bank.bank_version,
        )
        matches = schedule_matches(units)
        assert validate_schedule(matches, bank) == []
        for unit in units:
            game_a, game_b = unit.matches
            assert game_a.candidate_color == RED
            assert game_b.candidate_color == BLUE
            assert game_a.resolve_setups(bank) == game_b.resolve_setups(bank)
            assert game_a.paired_unit_id == game_b.paired_unit_id
            assert game_a.match_id != game_b.match_id
