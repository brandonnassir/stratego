"""The Phase 16 measurement contract: seeds, identities, board ids."""

import pytest

from stratego.evaluation.phase16 import contract
from stratego.evaluation.phase16.contract import (
    ADVERSARIAL_FAMILIES,
    AUTHORED_FAMILIES,
    BASELINE_ARMS,
    BENCHMARK_BASELINES,
    FAMILY_OPERATOR_HARVEST,
    MATCH_OPPONENTS,
    Phase16MeasurementError,
    SETUPS_PER_FAMILY,
    adversarial_board_id,
    benchmark_board_id,
    derive_measure_seed,
    parse_adversarial_board_id,
    parse_benchmark_board_id,
)


class TestSeeds:
    def test_deterministic(self):
        first = derive_measure_seed("measure_match", "board", 7)
        second = derive_measure_seed("measure_match", "board", 7)
        assert first == second

    def test_domains_separate(self):
        seeds = {
            domain: derive_measure_seed(domain, "same", 1)
            for domain in contract.STREAM_DOMAINS
        }
        assert len(set(seeds.values())) == len(seeds)

    def test_parts_separate(self):
        assert derive_measure_seed("measure_match", "a", 1) != derive_measure_seed(
            "measure_match", "a", 2
        )

    def test_distinct_from_phase15_streams(self):
        """Same payload text, different personalization — different stream."""
        from stratego.search.phase15.contract import derive_search_seed

        assert derive_measure_seed("measure_match", "x") != derive_search_seed(
            "search_match", "x"
        )

    def test_unknown_domain_refused(self):
        with pytest.raises(Phase16MeasurementError):
            derive_measure_seed("nope", 1)

    def test_colon_refused(self):
        with pytest.raises(Phase16MeasurementError):
            derive_measure_seed("measure_match", "a:b")

    def test_bool_refused(self):
        with pytest.raises(Phase16MeasurementError):
            derive_measure_seed("measure_match", True)


class TestBenchmarkBoardIds:
    def test_round_trip(self):
        identifier = benchmark_board_id("p24", "neutral_v1", "any", "red", 3)
        fields = parse_benchmark_board_id(identifier)
        assert fields == {
            "opponent": "p24",
            "setup_source": "neutral_v1",
            "family_key": "any",
            "color": "red",
            "ordinal": 3,
        }

    def test_unknown_opponent_refused(self):
        with pytest.raises(Phase16MeasurementError):
            benchmark_board_id("gary", "neutral_v1", "any", "red", 0)

    def test_unknown_source_refused(self):
        with pytest.raises(Phase16MeasurementError):
            benchmark_board_id("p24", "phase12_glue", "any", "red", 0)

    def test_bad_ordinal_refused(self):
        with pytest.raises(Phase16MeasurementError):
            benchmark_board_id("p24", "neutral_v1", "any", "red", -1)
        with pytest.raises(Phase16MeasurementError):
            benchmark_board_id("p24", "neutral_v1", "any", "red", 1000)

    def test_malformed_refused(self):
        with pytest.raises(Phase16MeasurementError):
            parse_benchmark_board_id("phase15_match_pack_v1|ms=1|opp=p24")


class TestAdversarialBoardIds:
    def test_round_trip(self):
        identifier = adversarial_board_id(
            "adversarial_opponent", "scout_screen", "p18", "blue", 41
        )
        fields = parse_adversarial_board_id(identifier)
        assert fields == {
            "arm": "adversarial_opponent",
            "family": "scout_screen",
            "opponent": "p18",
            "color": "blue",
            "pair_index": 41,
        }

    def test_unknown_arm_refused(self):
        with pytest.raises(Phase16MeasurementError):
            adversarial_board_id("arm4", "scout_screen", "p18", "blue", 0)

    def test_unknown_family_refused(self):
        with pytest.raises(Phase16MeasurementError):
            adversarial_board_id("benchmark_control", "surprise", "p18", "blue", 0)


class TestDesignConstants:
    def test_family_roster(self):
        assert FAMILY_OPERATOR_HARVEST in ADVERSARIAL_FAMILIES
        assert FAMILY_OPERATOR_HARVEST not in AUTHORED_FAMILIES
        assert len(AUTHORED_FAMILIES) == 8
        # 8 authored families x 12 setups = 96, inside the required 96-128.
        assert 96 <= len(AUTHORED_FAMILIES) * SETUPS_PER_FAMILY <= 128

    def test_arms(self):
        assert BASELINE_ARMS == (
            "benchmark_control",
            "adversarial_opponent",
            "adversarial_both",
        )

    def test_baselines_named_by_brief(self):
        assert ("p24_direct", "direct") in BENCHMARK_BASELINES
        assert ("p24_b24", "TINY") in BENCHMARK_BASELINES
        assert ("p24_b24", "MEDIUM") in BENCHMARK_BASELINES

    def test_opponent_roster_is_phase15_stage_b(self):
        assert len(MATCH_OPPONENTS) == 10
        assert MATCH_OPPONENTS[0] == "p18"
        assert "phase9_anchor" in MATCH_OPPONENTS
