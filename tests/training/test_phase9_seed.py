"""Regression: the frozen Phase 9 seeds, identities, and stream derivations.

Every value here was frozen by Agent 1 before any Phase 9 rollout, pilot, or
model result existed. A failing test means the frozen identity layer drifted,
which is a new contract version after review, never an in-place fix.
"""

import pytest

from stratego.training import phase9_seed as ps
from stratego.training import warmstart_seed as ws


class TestCanonicalSeeds:
    def test_the_eight_frozen_seed_values(self):
        assert ps.CANONICAL_PHASE9_SEEDS == {
            "phase9_master_seed": 2026081601,
            "rollout_schedule_seed": 2026081602,
            "opponent_schedule_seed": 2026081603,
            "train_order_seed": 2026081604,
            "pilot_namespace_seed": 2026081605,
            "canonical_namespace_seed": 2026081606,
            "validation_bootstrap_seed": 2026081607,
            "test_bootstrap_seed": 2026081608,
        }

    def test_module_constants_agree_with_the_table(self):
        assert ps.PHASE9_MASTER_SEED == 2026081601
        assert ps.ROLLOUT_SCHEDULE_SEED == 2026081602
        assert ps.OPPONENT_SCHEDULE_SEED == 2026081603
        assert ps.TRAIN_ORDER_SEED == 2026081604
        assert ps.PILOT_NAMESPACE_SEED == 2026081605
        assert ps.CANONICAL_NAMESPACE_SEED == 2026081606
        assert ps.VALIDATION_BOOTSTRAP_SEED == 2026081607
        assert ps.TEST_BOOTSTRAP_SEED == 2026081608

    def test_seeds_are_disjoint_from_phase_8(self):
        assert not set(ps.CANONICAL_PHASE9_SEEDS.values()) & set(
            ws.CANONICAL_SEEDS.values()
        )

    def test_bootstrap_seed_lookup(self):
        assert ps.bootstrap_seed("validation") == 2026081607
        assert ps.bootstrap_seed("test") == 2026081608
        with pytest.raises(ps.Phase9SeedError):
            ps.bootstrap_seed("train")

    def test_namespace_seed_lookup(self):
        assert ps.namespace_seed("canonical") == ps.CANONICAL_NAMESPACE_SEED
        for namespace in ps.PILOT_NAMESPACES:
            assert ps.namespace_seed(namespace) == ps.PILOT_NAMESPACE_SEED
        with pytest.raises(ps.Phase9SeedError):
            ps.namespace_seed("adhoc")


class TestDomainSeparation:
    def test_unknown_domain_is_refused(self):
        with pytest.raises(ps.Phase9SeedError):
            ps.derive_phase9_seed("not_a_domain", 1)

    def test_non_scalar_parts_are_refused(self):
        with pytest.raises(ps.Phase9SeedError):
            ps.derive_phase9_seed(ps.DOMAIN_SETUP_ROOT, [1])
        with pytest.raises(ps.Phase9SeedError):
            ps.derive_phase9_seed(ps.DOMAIN_SETUP_ROOT, True)

    def test_domains_are_separated(self):
        seeds = {
            ps.derive_phase9_seed(domain, "probe", 7) for domain in ps.STREAM_DOMAINS
        }
        assert len(seeds) == len(ps.STREAM_DOMAINS)

    def test_identity_stability(self):
        first = ps.derive_phase9_seed(ps.DOMAIN_SETUP_ROOT, "probe", 7)
        assert first == ps.derive_phase9_seed(ps.DOMAIN_SETUP_ROOT, "probe", 7)

    def test_any_identity_change_moves_the_stream(self):
        base = ps.derive_phase9_seed(ps.DOMAIN_TRAIN_ORDER, "a", 1, 2)
        assert base != ps.derive_phase9_seed(ps.DOMAIN_TRAIN_ORDER, "a", 1, 3)
        assert base != ps.derive_phase9_seed(ps.DOMAIN_TRAIN_ORDER, "a", 2, 2)
        assert base != ps.derive_phase9_seed(ps.DOMAIN_TRAIN_ORDER, "b", 1, 2)

    def test_phase9_streams_are_disjoint_from_phase_8_streams(self):
        """Same domain name + parts under the two personalizations differ."""
        phase9 = ps.derive_phase9_seed(ps.DOMAIN_SETUP_ROOT, "shared_probe")
        phase8 = ws.derive_warmstart_seed("setup_root", "shared_probe")
        assert phase9 != phase8

    def test_63_bit_range(self):
        seed = ps.derive_phase9_seed(ps.DOMAIN_SETUP_ROOT, "range_probe")
        assert 0 <= seed < 2**63


class TestGameIdentity:
    def test_format_example(self):
        game_id = ps.phase9_game_id("canonical", 12, "historical", 137)
        assert game_id == (
            "phase9_rollout_v1|ms=2026081601|ns=canonical|it=012|b=historical|g=0137"
        )

    def test_round_trip(self):
        for namespace in ("canonical", "pilot_p9c"):
            for bucket in ps.POPULATION_BUCKETS:
                game_id = ps.phase9_game_id(namespace, 60, bucket, 0)
                fields = ps.parse_phase9_game_id(game_id)
                assert fields == {
                    "rollout_version": "phase9_rollout_v1",
                    "phase9_master_seed": 2026081601,
                    "namespace": namespace,
                    "iteration": 60,
                    "bucket": bucket,
                    "ordinal": 0,
                }

    def test_rejects_bad_components(self):
        with pytest.raises(ps.Phase9SeedError):
            ps.phase9_game_id("adhoc", 1, "current", 0)
        with pytest.raises(ps.Phase9SeedError):
            ps.phase9_game_id("canonical", 1, "teacher", 0)
        with pytest.raises(ps.Phase9SeedError):
            ps.phase9_game_id("canonical", 0, "current", 0)
        with pytest.raises(ps.Phase9SeedError):
            ps.phase9_game_id("canonical", 1, "current", -1)
        with pytest.raises(ps.Phase9SeedError):
            ps.phase9_game_id("canonical", True, "current", 0)

    def test_rejects_foreign_and_tampered_ids(self):
        with pytest.raises(ps.Phase9SeedError):
            ps.parse_phase9_game_id("not a game id")
        with pytest.raises(ps.Phase9SeedError):
            ps.parse_phase9_game_id(
                "phase9_rollout_v1|ms=999|ns=canonical|it=001|b=current|g=0000"
            )
        with pytest.raises(ps.Phase9SeedError):
            ps.parse_phase9_game_id(
                "phase9_rollout_v2|ms=2026081601|ns=canonical|it=001|b=current|g=0000"
            )
        with pytest.raises(ps.Phase9SeedError):
            ps.parse_phase9_game_id(
                ws.synthetic_game_id("train", "random_legal@1.0.0", "random_legal@1.0.0", 0)
            )

    def test_identifier_uniqueness_across_a_sweep(self):
        identifiers = set()
        for iteration in (1, 2, 60):
            for bucket in ps.POPULATION_BUCKETS:
                for ordinal in range(3):
                    identifiers.add(
                        ps.phase9_game_id("canonical", iteration, bucket, ordinal)
                    )
                    identifiers.add(
                        ps.phase9_game_id("pilot_p9a", iteration, bucket, ordinal)
                    )
        assert len(identifiers) == 3 * len(ps.POPULATION_BUCKETS) * 3 * 2


class TestPerGameStreams:
    def test_game_seed_bundle(self):
        game_id = ps.phase9_game_id("canonical", 3, "historical", 4)
        seeds = ps.game_seeds(game_id)
        assert set(seeds) == {
            "setup_root_seed",
            "red_policy_seed",
            "blue_policy_seed",
            "historical_opponent_seed",
        }
        assert len(set(seeds.values())) == 4

    def test_non_historical_games_have_no_archive_stream(self):
        game_id = ps.phase9_game_id("canonical", 3, "rule", 4)
        seeds = ps.game_seeds(game_id)
        assert "historical_opponent_seed" not in seeds
        with pytest.raises(ps.Phase9SeedError):
            ps.historical_opponent_seed(game_id)

    def test_streams_require_a_valid_game_id(self):
        with pytest.raises(ps.Phase9SeedError):
            ps.setup_root_seed("garbage")

    def test_behavior_sampler_stream(self):
        game_id = ps.phase9_game_id("canonical", 1, "current", 0)
        assert ps.behavior_sample_seed(game_id, 0) != ps.behavior_sample_seed(game_id, 1)
        assert ps.behavior_sample_seed(game_id, 5) == ps.behavior_sample_seed(game_id, 5)
        uniform = ps.behavior_sample_uniform(game_id, 9)
        assert 0.0 < uniform <= 1.0
        with pytest.raises(ps.Phase9SeedError):
            ps.behavior_sample_seed(game_id, -1)

    def test_train_order_streams(self):
        seeds = {
            ps.train_order_seed(namespace, iteration, epoch)
            for namespace in ("canonical", "pilot_p9a", "pilot_p9b")
            for iteration in (1, 2)
            for epoch in (0, 1)
        }
        assert len(seeds) == 12
        with pytest.raises(ps.Phase9SeedError):
            ps.train_order_seed("canonical", 0, 0)
        with pytest.raises(ps.Phase9SeedError):
            ps.train_order_seed("canonical", 1, -1)

    def test_eval_bank_streams(self):
        seeds = {
            ps.eval_bank_draw_seed(bank, family, case, side, attempt)
            for bank in ("phase9_validation_bank_v1", "phase9_test_bank_v1")
            for family in ("F00", "F15")
            for case in (0, 7)
            for side in ("red", "blue")
            for attempt in (0, 1)
        }
        assert len(seeds) == 32
        with pytest.raises(ps.Phase9SeedError):
            ps.eval_bank_draw_seed("bank", "F00", 0, "green", 0)
        with pytest.raises(ps.Phase9SeedError):
            ps.eval_bank_draw_seed("bank", "F00", -1, "red", 0)
        with pytest.raises(ps.Phase9SeedError):
            ps.eval_bank_draw_seed("bank", "F00", 0, "red", -1)
