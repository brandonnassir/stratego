"""The sampled-rollout engine: mechanics for any weights (hermetic)."""

import pytest

from stratego.search.phase16.contract import (
    ROLLOUT_SEARCH_VERSION,
    Phase16StochasticError,
)
from stratego.search.phase16.engine import Phase16StochasticEngine
from stratego.search.phase16.stochastic import (
    StochasticArm,
    build_stochastic_bundle,
)


@pytest.fixture(scope="module")
def frozen_bundle(fake_models):
    return build_stochastic_bundle(
        fake_models, StochasticArm(0.0, 0.0), "TINY", device="cpu"
    )


@pytest.fixture(scope="module")
def sampled_bundle(fake_models):
    return build_stochastic_bundle(
        fake_models, StochasticArm(0.0, 1.0), "TINY", device="cpu"
    )


SEED = 424242


class TestConstruction:
    def test_tau_r_zero_is_the_accepted_engine_object(self, frozen_bundle):
        # Not a copy and not a subclass: the accepted builder's own engine.
        assert type(frozen_bundle.engine).__name__ == "Phase12SearchEngine"

    def test_tau_r_positive_is_the_stochastic_engine(self, sampled_bundle):
        assert isinstance(sampled_bundle.engine, Phase16StochasticEngine)
        assert sampled_bundle.engine.rollout_temperature == 1.0
        assert sampled_bundle.engine.rollout_top_p == 0.9

    def test_configuration_is_byte_identical_to_the_preset(
        self, frozen_bundle, sampled_bundle
    ):
        for field in (
            "worlds",
            "rollout_depth",
            "max_root_candidates",
            "beta",
            "epsilon",
            "deduplicate_worlds",
            "verify_world_public_surface",
            "production",
        ):
            assert getattr(sampled_bundle.config, field) == getattr(
                frozen_bundle.config, field
            )

    def test_bad_temperatures_refused(self, fake_models):
        with pytest.raises(Phase16StochasticError):
            StochasticArm(-0.1, 0.0)
        with pytest.raises(Phase16StochasticError):
            StochasticArm(0.0, -1.0)
        with pytest.raises(Phase16StochasticError):
            StochasticArm(0.0, 1.0, top_p=0.0)
        with pytest.raises(Phase16StochasticError):
            StochasticArm(0.0, 1.0, top_p=1.5)

    def test_oracle_pairing_refused(self):
        with pytest.raises(Phase16StochasticError):
            StochasticArm(0.15, 0.0, pairing_id="p24_oracle")

    def test_direct_pairing_refused(self):
        with pytest.raises(Phase16StochasticError):
            StochasticArm(0.15, 0.0, pairing_id="p24_direct")


class TestDecisions:
    def test_same_seeds_reproduce_exactly(self, sampled_bundle, midgame_state):
        first = sampled_bundle.engine.choose_action(
            midgame_state, seed=SEED, rollout_seed=7
        )
        again = sampled_bundle.engine.choose_action(
            midgame_state, seed=SEED, rollout_seed=7
        )
        assert first.selected_action_id == again.selected_action_id
        assert first.world_weights == again.world_weights
        assert [c.q_value for c in first.candidates] == [
            c.q_value for c in again.candidates
        ]
        assert [c.world_values for c in first.candidates] == [
            c.world_values for c in again.candidates
        ]

    def test_worlds_and_candidates_match_the_frozen_engine(
        self, frozen_bundle, sampled_bundle, midgame_state
    ):
        frozen = frozen_bundle.engine.choose_action(midgame_state, seed=SEED)
        sampled = sampled_bundle.engine.choose_action(
            midgame_state, seed=SEED, rollout_seed=7
        )
        # World sampling is untouched: same worlds, same multiplicities.
        assert sampled.world_weights == frozen.world_weights
        assert sampled.unique_worlds == frozen.unique_worlds
        # The candidate rule is untouched: same actions, same priors, same order.
        assert [c.absolute_action_id for c in sampled.candidates] == [
            c.absolute_action_id for c in frozen.candidates
        ]
        assert [c.prior for c in sampled.candidates] == [
            c.prior for c in frozen.candidates
        ]
        assert sampled.direct_action_id == frozen.direct_action_id

    def test_identity_stamp(self, frozen_bundle, sampled_bundle, midgame_state):
        frozen = frozen_bundle.engine.choose_action(midgame_state, seed=SEED)
        sampled = sampled_bundle.engine.choose_action(
            midgame_state, seed=SEED, rollout_seed=7
        )
        assert frozen.search_version == "phase12_root_world_search_v1"
        assert sampled.search_version == ROLLOUT_SEARCH_VERSION

    def test_selection_is_legal(self, sampled_bundle, midgame_state):
        from stratego.engine.legal_moves import legal_actions

        decision = sampled_bundle.engine.choose_action(
            midgame_state, seed=SEED, rollout_seed=3
        )
        assert decision.selected_action_id in set(legal_actions(midgame_state))

    def test_default_rollout_seed_is_deterministic(self, sampled_bundle, midgame_state):
        first = sampled_bundle.engine.choose_action(midgame_state, seed=SEED)
        again = sampled_bundle.engine.choose_action(midgame_state, seed=SEED)
        assert first.selected_action_id == again.selected_action_id
        assert [c.q_value for c in first.candidates] == [
            c.q_value for c in again.candidates
        ]

    def test_tau_r_zero_engine_delegates_bit_identically(
        self, fake_models, frozen_bundle, midgame_state
    ):
        # The class itself at tau_r = 0 must equal the accepted engine.
        engine = Phase16StochasticEngine(
            frozen_bundle.engine.model,
            frozen_bundle.engine.provider,
            frozen_bundle.config,
            rollout_temperature=0.0,
        )
        ours = engine.choose_action(midgame_state, seed=SEED)
        accepted = frozen_bundle.engine.choose_action(midgame_state, seed=SEED)
        assert ours.selected_action_id == accepted.selected_action_id
        assert ours.world_weights == accepted.world_weights
        assert [c.q_value for c in ours.candidates] == [
            c.q_value for c in accepted.candidates
        ]
        assert ours.search_version == accepted.search_version

    def test_describe_names_the_sampling(self, sampled_bundle):
        report = sampled_bundle.engine.describe()
        assert report["search_version"] == ROLLOUT_SEARCH_VERSION
        assert "sampled" in report["rollout_policy"]
        assert report["rollout_temperature"] == 1.0
