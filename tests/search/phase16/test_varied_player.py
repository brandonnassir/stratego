"""The varied player: modes, sampling, the never-forfeit chain (hermetic)."""

import pytest

from stratego.search.phase16.contract import (
    MODE_VARIED_FAST,
    MODE_VARIED_STRENGTH,
    Phase16StochasticError,
)
from stratego.search.phase16.stochastic import (
    Phase16VariedPlayer,
    StochasticArm,
    build_stochastic_bundle,
)


@pytest.fixture(scope="module")
def player(fake_models):
    arm = StochasticArm(0.15, 0.0)
    # TINY for both modes keeps the hermetic tests fast; the mode table and
    # decision mechanics are preset-independent.
    bundles = {
        MODE_VARIED_STRENGTH: build_stochastic_bundle(fake_models, arm, "TINY"),
        MODE_VARIED_FAST: build_stochastic_bundle(fake_models, arm, "TINY"),
    }
    return Phase16VariedPlayer(arm, bundles, fake_models)


class TestModes:
    def test_oracle_refused_by_name(self, player):
        for name in ("oracle", "p24_oracle"):
            with pytest.raises(Phase16StochasticError):
                player.check_mode(name)

    def test_unknown_mode_refused(self, player):
        with pytest.raises(Phase16StochasticError):
            player.check_mode("selected_search")  # a Phase 15 mode, not a varied one

    def test_diagnostic_bundle_refused(self, fake_models):
        from stratego.search.phase15.systems import build_engine

        arm = StochasticArm(0.15, 0.0)
        oracle_bundle = build_engine("p24_oracle", fake_models, "TINY", production=False)
        with pytest.raises(Phase16StochasticError):
            Phase16VariedPlayer(
                arm,
                {MODE_VARIED_STRENGTH: oracle_bundle, MODE_VARIED_FAST: oracle_bundle},
                fake_models,
            )

    def test_missing_mode_refused(self, fake_models):
        arm = StochasticArm(0.15, 0.0)
        bundle = build_stochastic_bundle(fake_models, arm, "TINY")
        with pytest.raises(Phase16StochasticError):
            Phase16VariedPlayer(arm, {MODE_VARIED_STRENGTH: bundle}, fake_models)


class TestDecide:
    def test_decision_is_legal_and_reproducible(self, player, midgame_state):
        from stratego.engine.legal_moves import legal_actions

        first = player.decide(midgame_state, mode=MODE_VARIED_FAST, game_id="g1")
        again = player.decide(midgame_state, mode=MODE_VARIED_FAST, game_id="g1")
        assert first.action_id in set(legal_actions(midgame_state))
        assert first.action_id == again.action_id  # same seeds, same draw
        assert first.searched and again.searched
        assert first.fallback_reason is None
        assert first.seed == again.seed
        assert first.move_seed == again.move_seed

    def test_game_id_changes_the_draw_streams(self, player, midgame_state):
        first = player.decide(midgame_state, mode=MODE_VARIED_FAST, game_id="g1")
        other = player.decide(midgame_state, mode=MODE_VARIED_FAST, game_id="g2")
        assert first.seed != other.seed
        assert first.move_seed != other.move_seed

    def test_timeout_falls_back_to_direct_and_never_forfeits(
        self, fake_models, midgame_state
    ):
        from stratego.engine.legal_moves import legal_actions

        arm = StochasticArm(0.15, 0.0)
        bundles = {
            MODE_VARIED_STRENGTH: build_stochastic_bundle(fake_models, arm, "TINY"),
            MODE_VARIED_FAST: build_stochastic_bundle(fake_models, arm, "TINY"),
        }
        capped = Phase16VariedPlayer(
            arm,
            bundles,
            fake_models,
            time_caps={MODE_VARIED_FAST: 1e-9, MODE_VARIED_STRENGTH: 1e-9},
        )
        decision = capped.decide(midgame_state, mode=MODE_VARIED_FAST, game_id="g1")
        assert decision.fallback_reason == "timeout"
        assert decision.searched is False
        assert decision.action_id in set(legal_actions(midgame_state))
        assert capped.fallback_counts.get("timeout") == 1

    def test_status_counts(self, player):
        status = player.status()
        assert status["decisions"] >= 3
        assert "sampled_move_changes" in status

    def test_describe_names_both_modes_and_refuses_oracle(self, player):
        report = player.describe()
        assert set(report["modes"]) == {MODE_VARIED_STRENGTH, MODE_VARIED_FAST}
        assert report["oracle_available_in_production"] is False
        assert report["mode_presets"] == {
            MODE_VARIED_STRENGTH: "MEDIUM",
            MODE_VARIED_FAST: "TINY",
        }
