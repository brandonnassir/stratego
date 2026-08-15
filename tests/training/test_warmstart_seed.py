"""Phase 8 Agent 1: corpus identity, canonical seeds, decision sampler.

Every frozen value asserted here is a regression anchor: Agents 2-7 derive
corpora, training order and statistics from these exact numbers, so a change
that slips past this file would silently re-identify the whole phase.
"""

import random

import pytest

from stratego.training import warmstart_seed as ws

RED_TOKEN = "strategic_rule_based@1.1.0"
BLUE_TOKEN = "random_legal@1.0.0"


def game_id(split="train", red=RED_TOKEN, blue=BLUE_TOKEN, ordinal=0):
    return ws.synthetic_game_id(split, red, blue, ordinal)


class TestCanonicalSeeds:
    def test_the_seven_frozen_seeds_hold_their_recorded_values(self):
        assert ws.CANONICAL_SEEDS == {
            "corpus_master_seed": 2026081301,
            "canonical_c1_init_seed": 2026081302,
            "train_order_seed": 2026081303,
            "pilot_namespace_seed": 2026081304,
            "final_run_namespace_seed": 2026081305,
            "validation_bootstrap_seed": 2026081306,
            "test_bootstrap_seed": 2026081307,
        }

    def test_the_seeds_are_pairwise_distinct(self):
        values = list(ws.CANONICAL_SEEDS.values())
        assert len(set(values)) == len(values)

    def test_module_constants_agree_with_the_mapping(self):
        assert ws.CORPUS_MASTER_SEED == ws.CANONICAL_SEEDS["corpus_master_seed"]
        assert ws.CANONICAL_C1_INIT_SEED == ws.CANONICAL_SEEDS["canonical_c1_init_seed"]
        assert ws.TRAIN_ORDER_SEED == ws.CANONICAL_SEEDS["train_order_seed"]
        assert ws.PILOT_NAMESPACE_SEED == ws.CANONICAL_SEEDS["pilot_namespace_seed"]
        assert ws.FINAL_RUN_NAMESPACE_SEED == ws.CANONICAL_SEEDS["final_run_namespace_seed"]
        assert ws.VALIDATION_BOOTSTRAP_SEED == ws.CANONICAL_SEEDS["validation_bootstrap_seed"]
        assert ws.TEST_BOOTSTRAP_SEED == ws.CANONICAL_SEEDS["test_bootstrap_seed"]

    def test_the_corpus_schedule_constants_are_frozen(self):
        assert ws.SYNTHETIC_CORPUS_VERSION == "synthetic_warmstart_corpus_v1"
        assert ws.DECISION_SAMPLER_VERSION == "warmstart_decision_sampler_v1"
        assert ws.CORPUS_SPLITS == ("train", "validation", "test")
        assert ws.GAMES_PER_CELL == {"train": 200, "validation": 40, "test": 40}
        assert ws.MAX_DECISIONS_PER_GAME == 64


class TestDeriveWarmstartSeed:
    def test_equal_identity_equal_seed(self):
        first = ws.derive_warmstart_seed(ws.DOMAIN_SETUP_ROOT, "a", 1)
        second = ws.derive_warmstart_seed(ws.DOMAIN_SETUP_ROOT, "a", 1)
        assert first == second

    def test_seeds_are_63_bit_non_negative(self):
        for domain in ws.STREAM_DOMAINS:
            seed = ws.derive_warmstart_seed(domain, "probe")
            assert 0 <= seed < 2**63

    def test_domains_separate_streams(self):
        seeds = {
            domain: ws.derive_warmstart_seed(domain, "same", "parts")
            for domain in ws.STREAM_DOMAINS
        }
        assert len(set(seeds.values())) == len(ws.STREAM_DOMAINS)

    def test_any_part_change_changes_the_stream(self):
        base = ws.derive_warmstart_seed(ws.DOMAIN_PILOT, "cand", "init")
        assert base != ws.derive_warmstart_seed(ws.DOMAIN_PILOT, "cand", "init2")
        assert base != ws.derive_warmstart_seed(ws.DOMAIN_PILOT, "cand2", "init")

    def test_unknown_domain_is_rejected(self):
        with pytest.raises(ws.WarmstartSeedError):
            ws.derive_warmstart_seed("not_a_domain", "x")

    def test_non_scalar_parts_are_rejected(self):
        with pytest.raises(ws.WarmstartSeedError):
            ws.derive_warmstart_seed(ws.DOMAIN_PILOT, ["list"])
        with pytest.raises(ws.WarmstartSeedError):
            ws.derive_warmstart_seed(ws.DOMAIN_PILOT, True)

    def test_no_global_rng_involvement(self):
        state = random.getstate()
        random.seed(1)
        first = ws.derive_warmstart_seed(ws.DOMAIN_SETUP_ROOT, "g")
        random.seed(999)
        second = ws.derive_warmstart_seed(ws.DOMAIN_SETUP_ROOT, "g")
        random.setstate(state)
        assert first == second


class TestSyntheticGameId:
    def test_format_is_the_frozen_parseable_layout(self):
        assert game_id(ordinal=137) == (
            "synthetic_warmstart_corpus_v1|ms=2026081301|split=train"
            "|red=strategic_rule_based@1.1.0|blue=random_legal@1.0.0|g=0137"
        )

    def test_round_trip_parse(self):
        parsed = ws.parse_synthetic_game_id(game_id("validation", ordinal=39))
        assert parsed == {
            "corpus_version": "synthetic_warmstart_corpus_v1",
            "corpus_master_seed": 2026081301,
            "split": "validation",
            "red_token": RED_TOKEN,
            "blue_token": BLUE_TOKEN,
            "ordinal": 39,
        }

    def test_identity_is_a_pure_function_of_the_required_fields(self):
        assert game_id() == game_id()
        distinct = {
            game_id("train", ordinal=0),
            game_id("train", ordinal=1),
            game_id("validation", ordinal=0),
            game_id("test", ordinal=0),
            game_id("train", red=BLUE_TOKEN, blue=RED_TOKEN, ordinal=0),
        }
        assert len(distinct) == 5

    def test_ordinal_bounds_follow_the_split_schedule(self):
        assert game_id("train", ordinal=199)
        assert game_id("validation", ordinal=39)
        assert game_id("test", ordinal=39)
        with pytest.raises(ws.WarmstartSeedError):
            game_id("train", ordinal=200)
        with pytest.raises(ws.WarmstartSeedError):
            game_id("validation", ordinal=40)
        with pytest.raises(ws.WarmstartSeedError):
            game_id("test", ordinal=-1)

    def test_malformed_inputs_are_rejected(self):
        with pytest.raises(ws.WarmstartSeedError):
            game_id(split="production")
        with pytest.raises(ws.WarmstartSeedError):
            game_id(red="Strategic@1.1.0")
        with pytest.raises(ws.WarmstartSeedError):
            game_id(blue="random_legal")
        with pytest.raises(ws.WarmstartSeedError):
            game_id(blue="random|legal@1.0.0")

    def test_foreign_and_tampered_ids_do_not_parse(self):
        good = game_id()
        with pytest.raises(ws.WarmstartSeedError):
            ws.parse_synthetic_game_id(good.replace("ms=2026081301", "ms=1"))
        with pytest.raises(ws.WarmstartSeedError):
            ws.parse_synthetic_game_id(good.replace(ws.SYNTHETIC_CORPUS_VERSION, "corpus_v9"))
        with pytest.raises(ws.WarmstartSeedError):
            ws.parse_synthetic_game_id(good.replace("g=0000", "g=0200"))
        with pytest.raises(ws.WarmstartSeedError):
            ws.parse_synthetic_game_id("not an id at all")


class TestPerGameSeeds:
    def test_the_three_per_game_domains_disagree(self):
        identifier = game_id()
        seeds = ws.game_seeds(identifier)
        assert len(set(seeds.values())) == 3
        assert seeds["setup_root_seed"] == ws.setup_root_seed(identifier)
        assert seeds["red_policy_seed"] == ws.red_policy_seed(identifier)
        assert seeds["blue_policy_seed"] == ws.blue_policy_seed(identifier)

    def test_seeds_differ_between_games(self):
        first = ws.game_seeds(game_id(ordinal=0))
        second = ws.game_seeds(game_id(ordinal=1))
        for key in first:
            assert first[key] != second[key]

    def test_color_swapped_cells_receive_unrelated_streams(self):
        forward = ws.game_seeds(game_id(red=RED_TOKEN, blue=BLUE_TOKEN))
        reverse = ws.game_seeds(game_id(red=BLUE_TOKEN, blue=RED_TOKEN))
        assert forward["setup_root_seed"] != reverse["setup_root_seed"]
        assert forward["red_policy_seed"] != reverse["red_policy_seed"]

    def test_per_game_seeds_require_a_valid_game_id(self):
        with pytest.raises(ws.WarmstartSeedError):
            ws.setup_root_seed("bogus")
        with pytest.raises(ws.WarmstartSeedError):
            ws.red_policy_seed("bogus")
        with pytest.raises(ws.WarmstartSeedError):
            ws.blue_policy_seed("bogus")


class TestDecisionSampler:
    def test_zero_decision_game_selects_nothing(self):
        assert ws.selected_decision_indices(game_id(), 0) == ()

    def test_short_games_select_every_decision(self):
        for total in (1, 2, 63, 64):
            assert ws.selected_decision_indices(game_id(), total) == tuple(range(total))

    def test_negative_totals_are_rejected(self):
        with pytest.raises(ws.WarmstartSeedError):
            ws.selected_decision_indices(game_id(), -1)

    def test_bin_bounds_partition_the_index_range(self):
        for total in (65, 100, 128, 700, 4000):
            bounds = ws.decision_bin_bounds(total)
            assert len(bounds) == 64
            assert bounds[0][0] == 0
            assert bounds[-1][1] == total
            for (low, high), (next_low, _) in zip(bounds, bounds[1:]):
                assert high == next_low
                assert high > low
            assert all(high > low for low, high in bounds)

    def test_bin_bounds_are_undefined_at_or_below_the_cap(self):
        with pytest.raises(ws.WarmstartSeedError):
            ws.decision_bin_bounds(64)

    def test_long_games_select_one_index_per_bin_sorted(self):
        identifier = game_id(ordinal=7)
        for total in (65, 129, 700, 3999):
            selected = ws.selected_decision_indices(identifier, total)
            assert len(selected) == 64
            assert len(set(selected)) == 64
            assert all(0 <= index < total for index in selected)
            assert all(a < b for a, b in zip(selected, selected[1:]))
            for bin_index, (low, high) in enumerate(ws.decision_bin_bounds(total)):
                assert low <= selected[bin_index] < high

    def test_selection_reproduces_from_the_published_arithmetic(self):
        identifier = game_id(ordinal=11)
        total = 700
        expected = tuple(
            low + ws.decision_bin_seed(identifier, index) % (high - low)
            for index, (low, high) in enumerate(ws.decision_bin_bounds(total))
        )
        assert ws.selected_decision_indices(identifier, total) == expected

    def test_selection_is_deterministic_and_game_specific(self):
        first = ws.selected_decision_indices(game_id(ordinal=3), 700)
        again = ws.selected_decision_indices(game_id(ordinal=3), 700)
        other = ws.selected_decision_indices(game_id(ordinal=4), 700)
        assert first == again
        assert first != other

    def test_bin_seed_requires_a_bin_inside_the_cap(self):
        with pytest.raises(ws.WarmstartSeedError):
            ws.decision_bin_seed(game_id(), 64)
        with pytest.raises(ws.WarmstartSeedError):
            ws.decision_bin_seed(game_id(), -1)


class TestAuxiliaryStreams:
    def test_train_order_seeds_differ_per_epoch(self):
        assert ws.train_order_seed(0) != ws.train_order_seed(1)
        assert ws.train_order_seed(0) == ws.train_order_seed(0)
        with pytest.raises(ws.WarmstartSeedError):
            ws.train_order_seed(-1)

    def test_pilot_streams_are_namespaced_by_candidate_and_purpose(self):
        base = ws.pilot_stream_seed("ws_pilot_lr1e-3_balanced", "data_order")
        assert base == ws.pilot_stream_seed("ws_pilot_lr1e-3_balanced", "data_order")
        assert base != ws.pilot_stream_seed("ws_pilot_lr1e-3_policy_led", "data_order")
        assert base != ws.pilot_stream_seed("ws_pilot_lr1e-3_balanced", "other")
        with pytest.raises(ws.WarmstartSeedError):
            ws.pilot_stream_seed("", "data_order")

    def test_final_run_streams_are_disjoint_from_pilot_streams(self):
        assert ws.final_run_stream_seed("data_order") != ws.pilot_stream_seed(
            "ws_pilot_lr1e-3_balanced", "data_order"
        )
        with pytest.raises(ws.WarmstartSeedError):
            ws.final_run_stream_seed(" ")

    def test_bootstrap_seeds_cover_exactly_the_held_out_splits(self):
        assert ws.bootstrap_seed("validation") == 2026081306
        assert ws.bootstrap_seed("test") == 2026081307
        with pytest.raises(ws.WarmstartSeedError):
            ws.bootstrap_seed("train")
