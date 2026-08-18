"""Regression: the frozen Phase 10 seeds, identities, and stream derivations.

Every value here was frozen by Agent 1 before any Phase 10 outcome game was
played and before either utility model was fit. A failing test means the
frozen identity layer drifted, which is a new contract version after review,
never an in-place fix.
"""

import pytest

from stratego.setups.seed import DEFAULT_LIBRARY_MASTER_SEED
from stratego.training import phase10_seed as ps
from stratego.training import phase9_seed as p9
from stratego.training import warmstart_seed as ws


class TestCanonicalSeeds:
    def test_the_eight_frozen_seed_values(self):
        assert ps.CANONICAL_PHASE10_SEEDS == {
            "phase10_master_seed": 2026081801,
            "outcome_schedule_seed": 2026081802,
            "setup_draw_seed": 2026081803,
            "utility_fit_seed": 2026081804,
            "selector_draw_seed": 2026081805,
            "case_schedule_seed": 2026081806,
            "validation_bootstrap_seed": 2026081807,
            "test_bootstrap_seed": 2026081808,
        }

    def test_module_constants_agree_with_the_table(self):
        assert ps.PHASE10_MASTER_SEED == 2026081801
        assert ps.OUTCOME_SCHEDULE_SEED == 2026081802
        assert ps.SETUP_DRAW_SEED == 2026081803
        assert ps.UTILITY_FIT_SEED == 2026081804
        assert ps.SELECTOR_DRAW_SEED == 2026081805
        assert ps.CASE_SCHEDULE_SEED == 2026081806
        assert ps.VALIDATION_BOOTSTRAP_SEED == 2026081807
        assert ps.TEST_BOOTSTRAP_SEED == 2026081808

    def test_seeds_are_disjoint_from_every_accepted_upstream_block(self):
        phase10 = set(ps.CANONICAL_PHASE10_SEEDS.values())
        assert not phase10 & set(p9.CANONICAL_PHASE9_SEEDS.values())
        assert not phase10 & set(ws.CANONICAL_SEEDS.values())
        assert DEFAULT_LIBRARY_MASTER_SEED not in phase10

    def test_bootstrap_roots(self):
        assert ps.bootstrap_root("validation") == 2026081807
        assert ps.bootstrap_root("test") == 2026081808
        with pytest.raises(ps.Phase10SeedError):
            ps.bootstrap_root("train")

    def test_every_domain_names_a_frozen_root(self):
        assert set(ps.DOMAIN_ROOTS) == set(ps.STREAM_DOMAINS)
        roots = set(ps.CANONICAL_PHASE10_SEEDS.values())
        assert set(ps.DOMAIN_ROOTS.values()) <= roots


class TestDomainSeparation:
    def test_unknown_domain_is_refused(self):
        with pytest.raises(ps.Phase10SeedError):
            ps.derive_phase10_seed("not_a_domain", 1)

    def test_bool_identity_parts_are_refused(self):
        with pytest.raises(ps.Phase10SeedError):
            ps.derive_phase10_seed(ps.DOMAIN_CORPUS_MATCH, True)

    def test_equal_identities_agree_and_differing_ones_do_not(self):
        first = ps.derive_phase10_seed(ps.DOMAIN_CORPUS_MATCH, "a", 1)
        assert first == ps.derive_phase10_seed(ps.DOMAIN_CORPUS_MATCH, "a", 1)
        assert first != ps.derive_phase10_seed(ps.DOMAIN_CORPUS_MATCH, "a", 2)

    def test_domains_sharing_a_root_do_not_collide(self):
        assert ps.DOMAIN_ROOTS[ps.DOMAIN_SELECTOR_BRANCH] == ps.DOMAIN_ROOTS[
            ps.DOMAIN_SELECTOR_BASE
        ]
        parts = ("sel", "train", "red", 7)
        assert ps.derive_phase10_seed(
            ps.DOMAIN_SELECTOR_BRANCH, *parts
        ) != ps.derive_phase10_seed(ps.DOMAIN_SELECTOR_BASE, *parts)

    def test_seeds_are_63_bit_non_negative(self):
        seed = ps.derive_phase10_seed(ps.DOMAIN_CORPUS_MATCH, "x")
        assert 0 <= seed < 2**63


class TestOutcomeGameIdentity:
    def test_format_and_round_trip(self):
        game_id = ps.phase10_game_id("F03", "F11", 7)
        assert game_id == "phase10_outcome_v1|ms=2026081801|rf=F03|bf=F11|g=07"
        assert ps.parse_phase10_game_id(game_id) == {
            "outcome_version": "phase10_outcome_v1",
            "phase10_master_seed": 2026081801,
            "red_family": "F03",
            "blue_family": "F11",
            "ordinal": 7,
        }

    def test_ordering_matters(self):
        assert ps.phase10_game_id("F03", "F11", 0) != ps.phase10_game_id("F11", "F03", 0)

    @pytest.mark.parametrize(
        "arguments",
        [
            ("f03", "F11", 0),
            ("F3", "F11", 0),
            ("F03", "F111", 0),
            ("F03", "F11", -1),
            ("F03", "F11", 100),
        ],
    )
    def test_malformed_identities_are_refused(self, arguments):
        with pytest.raises(ps.Phase10SeedError):
            ps.phase10_game_id(*arguments)

    @pytest.mark.parametrize(
        "game_id",
        [
            "phase9_rollout_v1|ms=2026081601|ns=canonical|it=001|b=current|g=0001",
            "phase10_outcome_v2|ms=2026081801|rf=F03|bf=F11|g=07",
            "phase10_outcome_v1|ms=2026081802|rf=F03|bf=F11|g=07",
            "phase10_outcome_v1|ms=2026081801|rf=F03|bf=F11|g=7",
        ],
    )
    def test_foreign_ids_are_refused(self, game_id):
        with pytest.raises(ps.Phase10SeedError):
            ps.parse_phase10_game_id(game_id)

    def test_corpus_streams_are_identity_pure(self):
        game_id = ps.phase10_game_id("F00", "F00", 0)
        assert ps.corpus_setup_seed(game_id, "red", 0) != ps.corpus_setup_seed(
            game_id, "blue", 0
        )
        assert ps.corpus_setup_seed(game_id, "red", 0) != ps.corpus_setup_seed(
            game_id, "red", 1
        )
        assert ps.corpus_match_seed(game_id) == ps.corpus_match_seed(game_id)
        with pytest.raises(ps.Phase10SeedError):
            ps.corpus_setup_seed(game_id, "green", 0)


class TestCaseIdentity:
    def test_format_and_round_trip(self):
        case_id = ps.phase10_case_id("phase10_validation_bank_v1", "F03", 5)
        assert case_id == "phase10_validation_bank_v1|ms=2026081801|f=F03|c=005"
        assert ps.parse_phase10_case_id(case_id)["family_id"] == "F03"

    def test_the_two_banks_never_share_a_case_id(self):
        validation = ps.phase10_case_id("phase10_validation_bank_v1", "F03", 5)
        test = ps.phase10_case_id("phase10_test_bank_v1", "F03", 5)
        assert validation != test

    def test_bank_version_must_follow_the_naming_rule(self):
        with pytest.raises(ps.Phase10SeedError):
            ps.phase10_case_id("some_other_bank", "F03", 5)

    def test_selector_seed_is_per_colour_and_walks_attempts(self):
        case_id = ps.phase10_case_id("phase10_validation_bank_v1", "F00", 0)
        assert ps.case_selector_seed(case_id, "red") != ps.case_selector_seed(
            case_id, "blue"
        )
        assert ps.case_selector_seed(case_id, "red") == ps.case_selector_seed(
            case_id, "red", 0
        )
        assert ps.case_selector_seed(case_id, "red", 0) != ps.case_selector_seed(
            case_id, "red", 1
        )

    def test_match_seed_is_arm_and_candidate_independent(self):
        case_id = ps.phase10_case_id("phase10_test_bank_v1", "F07", 3)
        first = ps.case_match_seed(case_id, 0, "vs_strategic")
        assert first == ps.case_match_seed(case_id, 0, "vs_strategic")
        assert first != ps.case_match_seed(case_id, 1, "vs_strategic")
        assert first != ps.case_match_seed(case_id, 0, "vs_tactical")
        with pytest.raises(ps.Phase10SeedError):
            ps.case_match_seed(case_id, 2, "vs_strategic")

    def test_colour_pairing_is_frozen(self):
        assert ps.CASE_GAME_INDICES == (0, 1)
        assert ps.CASE_GAME_COLOR == {0: "red", 1: "blue"}


class TestSelectorStreams:
    def test_uniforms_are_in_the_unit_interval_and_deterministic(self):
        branch = ps.selector_branch_uniform("P10-A", "validation", "red", 12345)
        assert 0.0 <= branch < 1.0
        assert branch == ps.selector_branch_uniform("P10-A", "validation", "red", 12345)

    def test_branch_and_base_streams_are_independent(self):
        arguments = ("P10-A", "validation", "red", 12345)
        assert ps.selector_branch_uniform(*arguments) != ps.selector_base_uniform(
            *arguments
        )

    def test_colour_and_identity_change_the_draw(self):
        base = ps.selector_base_uniform("P10-A", "validation", "red", 1)
        assert base != ps.selector_base_uniform("P10-A", "validation", "blue", 1)
        assert base != ps.selector_base_uniform("P10-B", "validation", "red", 1)
        assert base != ps.selector_base_uniform("P10-A", "test", "red", 1)

    def test_negative_selector_seeds_are_refused(self):
        with pytest.raises(ps.Phase10SeedError):
            ps.selector_base_uniform("P10-A", "validation", "red", -1)


class TestBootstrapStreams:
    def test_tokens_and_banks_separate_streams(self):
        first = ps.bootstrap_stream_seed("validation", "vs_strategic:delta")
        assert first != ps.bootstrap_stream_seed("validation", "vs_tactical:delta")
        assert first != ps.bootstrap_stream_seed("test", "vs_strategic:delta")

    def test_empty_token_is_refused(self):
        with pytest.raises(ps.Phase10SeedError):
            ps.bootstrap_stream_seed("validation", "")


class TestCollisionAudit:
    def test_duplicates_inside_a_stream_are_reported(self):
        audit = ps.stream_collision_audit({"a": [1, 1, 2]})
        assert not audit["no_collisions"]

    def test_collisions_across_streams_are_reported(self):
        audit = ps.stream_collision_audit({"a": [1, 2], "b": [2, 3]})
        assert not audit["no_collisions"]

    def test_disjoint_streams_pass(self):
        audit = ps.stream_collision_audit({"a": [1, 2], "b": [3, 4]})
        assert audit["no_collisions"]
        assert audit["distinct_seeds"] == 4


class TestSeedDocument:
    def test_document_publishes_every_domain(self):
        document = ps.seed_derivation_document()
        assert set(document["domains"]) == set(ps.STREAM_DOMAINS)
        assert document["root_seeds"] == ps.CANONICAL_PHASE10_SEEDS
        assert document["personalization"] == "strat-s10"

    def test_document_records_the_root_reading_notes(self):
        notes = ps.seed_derivation_document()["root_reading_notes"]
        assert any("test-case root" in note for note in notes)


class TestSelectorAuditDraws:
    """Agent 4's 3.6M audit draws are addressable by draw id, not by counter."""

    def test_draw_id_format_and_round_trip(self):
        draw_id = ps.selector_audit_draw_id("P10-D", "validation", "red", 42)
        assert draw_id == (
            "phase10_selector_audit_v1|ms=2026081801|k=P10-D|s=validation|c=red|n=00042"
        )
        assert ps.parse_selector_audit_draw_id(draw_id) == {
            "phase10_master_seed": 2026081801,
            "candidate_id": "P10-D",
            "split": "validation",
            "color": "red",
            "draw_ordinal": 42,
        }

    def test_the_frozen_audit_volume_sorts_lexicographically(self):
        ordinals = [0, 9, 10, 99_999]
        identifiers = [
            ps.selector_audit_draw_id("P10-A", "train", "red", ordinal)
            for ordinal in ordinals
        ]
        assert identifiers == sorted(identifiers)

    @pytest.mark.parametrize(
        "arguments",
        [
            ("P10-G", "train", "red", 0),
            ("P10-A", "holdout", "red", 0),
            ("P10-A", "train", "green", 0),
            ("P10-A", "train", "red", -1),
        ],
    )
    def test_malformed_draw_identities_are_refused(self, arguments):
        with pytest.raises(ps.Phase10SeedError):
            ps.selector_audit_draw_id(*arguments)

    def test_consecutive_ordinals_receive_unrelated_streams(self):
        first = ps.selector_audit_seed("P10-A", "train", "red", 0)
        second = ps.selector_audit_seed("P10-A", "train", "red", 1)
        assert first != second
        assert abs(first - second) > 1

    def test_every_identity_input_separates_the_stream(self):
        base = ps.selector_audit_seed("P10-A", "train", "red", 7)
        assert base != ps.selector_audit_seed("P10-B", "train", "red", 7)
        assert base != ps.selector_audit_seed("P10-A", "validation", "red", 7)
        assert base != ps.selector_audit_seed("P10-A", "train", "blue", 7)

    def test_audit_seeds_never_collide_with_bank_case_seeds(self):
        case_seeds = set()
        for bank in ("phase10_validation_bank_v1", "phase10_test_bank_v1"):
            for ordinal in range(32):
                case_id = ps.phase10_case_id(bank, "F00", ordinal)
                for color in ps.COLORS:
                    case_seeds.add(ps.case_selector_seed(case_id, color, 0))
        audit_seeds = {
            ps.selector_audit_seed("P10-A", split, color, ordinal)
            for split in ("train", "validation", "test")
            for color in ps.COLORS
            for ordinal in range(512)
        }
        assert not case_seeds & audit_seeds

    def test_the_domain_hangs_off_an_existing_root(self):
        assert ps.DOMAIN_SELECTOR_AUDIT in ps.STREAM_DOMAINS
        assert ps.DOMAIN_ROOTS[ps.DOMAIN_SELECTOR_AUDIT] == ps.SELECTOR_DRAW_SEED
        assert len(ps.CANONICAL_PHASE10_SEEDS) == 8
        assert len(ps.STREAM_DOMAINS) == 10
