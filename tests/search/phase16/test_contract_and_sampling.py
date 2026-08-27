"""Phase 16 contract: tokens, seeds, grids, and the two sampling primitives."""

import numpy as np
import pytest

from stratego.search.phase15.contract import derive_search_seed as derive_p15
from stratego.search.phase16.contract import (
    CONTROL_ARM,
    DOMAIN_MOVE_SAMPLE,
    DOMAIN_ROLLOUT_SAMPLE,
    MOVE_TAUS,
    ROLLOUT_TAUS,
    ROLLOUT_TOP_P,
    Phase16StochasticError,
    arm_name,
    derive_stochastic_seed,
    grid_arms,
    move_sample_seed,
    parse_arm_name,
    rollout_sample_seed,
    tau_r_token,
    tau_token,
)
from stratego.search.phase16.engine import _sampled_model_action


class TestTokensAndArms:
    def test_tokens(self):
        assert tau_token(0.0) == "t000"
        assert tau_token(0.15) == "t015"
        assert tau_token(0.6) == "t060"
        assert tau_r_token(1.0) == "r100"

    def test_token_refuses_inexact(self):
        with pytest.raises(Phase16StochasticError):
            tau_token(0.1234)

    def test_arm_name_round_trip(self):
        for tau in MOVE_TAUS:
            for tau_r in ROLLOUT_TAUS:
                assert parse_arm_name(arm_name(tau, tau_r)) == (tau, tau_r)

    def test_parse_refuses_malformed(self):
        for bad in ("stoch_t15_r100", "oracle", "stoch_t015_r100x", ""):
            with pytest.raises(Phase16StochasticError):
                parse_arm_name(bad)

    def test_control_arm(self):
        assert CONTROL_ARM == "stoch_t000_r000"
        assert parse_arm_name(CONTROL_ARM) == (0.0, 0.0)

    def test_grid_shape(self):
        arms = grid_arms()
        assert len(arms) == len(MOVE_TAUS) * len(ROLLOUT_TAUS)
        assert (0.0, 0.0) in arms
        assert len(set(arms)) == len(arms)


class TestSeeds:
    def test_deterministic(self):
        first = derive_stochastic_seed(DOMAIN_MOVE_SAMPLE, "x", 1)
        assert first == derive_stochastic_seed(DOMAIN_MOVE_SAMPLE, "x", 1)
        assert 0 <= first < 2**63

    def test_domains_separate(self):
        assert derive_stochastic_seed(DOMAIN_MOVE_SAMPLE, "x", 1) != derive_stochastic_seed(
            DOMAIN_ROLLOUT_SAMPLE, "x", 1
        )

    def test_distinct_from_phase15_streams(self):
        # Same textual parts through the accepted Phase 15 derivation must not
        # coincide: the personalization separates the phases.
        p16 = derive_stochastic_seed(DOMAIN_MOVE_SAMPLE, "board", 12)
        p15 = derive_p15("search_worlds", "board", 12)
        assert p16 != p15

    def test_unknown_domain_refused(self):
        with pytest.raises(Phase16StochasticError):
            derive_stochastic_seed("search_worlds", "x")

    def test_colon_refused(self):
        with pytest.raises(Phase16StochasticError):
            derive_stochastic_seed(DOMAIN_MOVE_SAMPLE, "a:b")

    def test_replay_and_arm_separate_streams(self):
        base = move_sample_seed(0.15, 0.0, "pos", 12, 0)
        assert base != move_sample_seed(0.15, 0.0, "pos", 12, 1)
        assert base != move_sample_seed(0.3, 0.0, "pos", 12, 0)
        assert base != move_sample_seed(0.15, 1.0, "pos", 12, 0)

    def test_rollout_stream_ignores_move_tau_by_design(self):
        # Every move-temperature arm at the same tau_r shares searches.
        assert rollout_sample_seed(1.0, ROLLOUT_TOP_P, "pos", 12, 3) == rollout_sample_seed(
            1.0, ROLLOUT_TOP_P, "pos", 12, 3
        )
        assert rollout_sample_seed(1.0, ROLLOUT_TOP_P, "pos", 12, 3) != rollout_sample_seed(
            1.0, ROLLOUT_TOP_P, "pos", 12, 4
        )


class _FakeCandidate:
    def __init__(self, action, score):
        self.absolute_action_id = action
        self.score = score


class _FakeDecision:
    def __init__(self, actions, scores, selected):
        self.candidates = tuple(
            _FakeCandidate(action, score) for action, score in zip(actions, scores)
        )
        self.selected_action_id = selected


class TestMoveSampling:
    def test_tau_zero_returns_frozen_choice_and_uses_no_rng(self):
        from stratego.search.phase16.stochastic import sample_move

        decision = _FakeDecision([10, 20, 30], [0.5, 0.9, 0.1], selected=20)
        action, record = sample_move(decision, 0.0, None)
        assert action == 20
        assert record["sampled"] is False
        assert record["changed_from_argmax"] is False

    def test_reproducible_from_seed(self):
        from stratego.search.phase16.stochastic import sample_move

        decision = _FakeDecision([10, 20, 30], [0.5, 0.52, 0.48], selected=20)
        first = [
            sample_move(decision, 0.3, np.random.Generator(np.random.PCG64(9)))[0]
            for _ in range(1)
        ]
        second = [
            sample_move(decision, 0.3, np.random.Generator(np.random.PCG64(9)))[0]
            for _ in range(1)
        ]
        assert first == second

    def test_distribution_tracks_softmax(self):
        from stratego.search.phase16.stochastic import move_distribution, sample_move

        decision = _FakeDecision([10, 20, 30], [0.4, 0.3, 0.1], selected=10)
        actions, probabilities = move_distribution(decision, 0.15)
        rng = np.random.Generator(np.random.PCG64(123))
        counts = {action: 0 for action in actions}
        draws = 4000
        for _ in range(draws):
            action, _record = sample_move(decision, 0.15, rng)
            counts[action] += 1
        for action, probability in zip(actions, probabilities):
            assert counts[action] / draws == pytest.approx(probability, abs=0.03)

    def test_low_tau_concentrates_on_argmax(self):
        from stratego.search.phase16.stochastic import sample_move

        decision = _FakeDecision([10, 20, 30], [0.9, 0.3, 0.1], selected=10)
        rng = np.random.Generator(np.random.PCG64(5))
        actions = {sample_move(decision, 0.01, rng)[0] for _ in range(200)}
        assert actions == {10}

    def test_non_finite_scores_refused(self):
        from stratego.search.phase12.contract import Phase12SearchError
        from stratego.search.phase16.stochastic import sample_move

        decision = _FakeDecision([10, 20], [float("nan"), 0.1], selected=20)
        with pytest.raises(Phase12SearchError):
            sample_move(decision, 0.3, np.random.Generator(np.random.PCG64(1)))

    def test_tau_positive_requires_rng(self):
        from stratego.search.phase16.stochastic import sample_move

        decision = _FakeDecision([10, 20], [0.2, 0.1], selected=10)
        with pytest.raises(Phase16StochasticError):
            sample_move(decision, 0.3, None)


class TestRolloutSampling:
    def _row(self, logits_by_id, width=8):
        row = np.full(width, -50.0, dtype=np.float32)
        for action, logit in logits_by_id.items():
            row[action] = logit
        return row

    def test_nucleus_trims_the_tail(self):
        # legal probs ~ [0.5, 0.3, 0.15, 0.05]: top_p=0.9 keeps the first 3.
        logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
        row = self._row({1: logits[0], 3: logits[1], 5: logits[2], 7: logits[3]})
        legal = np.array([1, 3, 5, 7], dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(7))
        seen = {
            _sampled_model_action(rng, row, legal, 1.0, 0.9) for _ in range(2000)
        }
        assert seen == {1, 3, 5}  # the 0.05 tail is trimmed

    def test_top_p_one_keeps_everything(self):
        logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
        row = self._row({1: logits[0], 3: logits[1], 5: logits[2], 7: logits[3]})
        legal = np.array([1, 3, 5, 7], dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(7))
        seen = {
            _sampled_model_action(rng, row, legal, 1.0, 1.0) for _ in range(4000)
        }
        assert seen == {1, 3, 5, 7}

    def test_frequencies_track_renormalized_nucleus(self):
        logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
        row = self._row({1: logits[0], 3: logits[1], 5: logits[2], 7: logits[3]})
        legal = np.array([1, 3, 5, 7], dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(11))
        counts = {1: 0, 3: 0, 5: 0, 7: 0}
        draws = 5000
        for _ in range(draws):
            counts[_sampled_model_action(rng, row, legal, 1.0, 0.9)] += 1
        total = 0.5 + 0.3 + 0.15
        assert counts[1] / draws == pytest.approx(0.5 / total, abs=0.03)
        assert counts[3] / draws == pytest.approx(0.3 / total, abs=0.03)
        assert counts[5] / draws == pytest.approx(0.15 / total, abs=0.03)
        assert counts[7] == 0

    def test_low_temperature_is_effectively_greedy(self):
        row = self._row({2: 1.0, 4: 0.5, 6: 0.2})
        legal = np.array([2, 4, 6], dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(3))
        seen = {
            _sampled_model_action(rng, row, legal, 0.01, ROLLOUT_TOP_P)
            for _ in range(200)
        }
        assert seen == {2}

    def test_only_legal_actions_sampled(self):
        row = np.zeros(16, dtype=np.float32)  # illegal ids equally attractive
        legal = np.array([4, 9], dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(2))
        for _ in range(100):
            assert _sampled_model_action(rng, row, legal, 1.0, 1.0) in {4, 9}

    def test_non_finite_legal_logit_refused(self):
        from stratego.search.phase12.contract import Phase12SearchError

        row = self._row({1: np.nan, 3: 0.0})
        legal = np.array([1, 3], dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(1))
        with pytest.raises(Phase12SearchError):
            _sampled_model_action(rng, row, legal, 1.0, 0.9)

    def test_deterministic_given_generator_state(self):
        logits = np.log(np.array([0.4, 0.35, 0.25]))
        row = self._row({1: logits[0], 3: logits[1], 5: logits[2]})
        legal = np.array([1, 3, 5], dtype=np.int64)
        first = [
            _sampled_model_action(
                np.random.Generator(np.random.PCG64(42)), row, legal, 1.0, 0.9
            )
            for _ in range(5)
        ]
        second = [
            _sampled_model_action(
                np.random.Generator(np.random.PCG64(42)), row, legal, 1.0, 0.9
            )
            for _ in range(5)
        ]
        assert first == second
