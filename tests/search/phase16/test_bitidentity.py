"""The brief's non-negotiable regression: tau = 0 and tau_r = 0 replay the
frozen Phase 15 decisions bit-identically, on the delivered bytes.

These tests bind to the real frozen stack (they skip cleanly when the
Phase 15 handoff is absent) and compare against the *stored* Stage A rows in
`reports/phase15/agent_02_decisions.csv` — the decisions the accepted
`phase15_search_candidate_v1` selection was read from — not against a fresh
run of the same code, so a drift anywhere in the Phase 16 path shows up as a
mismatch with history, exactly as the brief requires.
"""

import pytest

from stratego.search.phase16.contract import DECISION_SEED
from stratego.search.phase16.stochastic import (
    StochasticArm,
    build_stochastic_bundle,
    sample_move,
)

#: How many stored positions the regression replays. Enough to exercise the
#: candidate rule, worlds, dedup and the margin bookkeeping; small enough to
#: keep the suite fast.
POSITIONS = 4


@pytest.fixture(scope="module")
def replayed(models, phase15_position_manifest):
    from stratego.search.phase15.positions import materialize_positions

    subset = dict(phase15_position_manifest)
    subset["positions"] = phase15_position_manifest["positions"][:POSITIONS]
    return materialize_positions(subset, verify=True)


@pytest.fixture(scope="module")
def stored_rows(phase15_stage_a_rows):
    return {
        (row["position_id"], row["arm_id"]): row
        for row in phase15_stage_a_rows
        if row["arm_id"] == "p24_b24" and row["preset_id"] == "TINY"
    }


class TestZeroTemperatureBitIdentity:
    def test_control_replays_the_frozen_stage_a_decisions(
        self, models, replayed, stored_rows
    ):
        bundle = build_stochastic_bundle(
            models, StochasticArm(0.0, 0.0), "TINY", device="cpu"
        )
        compared = 0
        for position, state, _plan in replayed:
            stored = stored_rows.get((position["position_id"], "p24_b24"))
            assert stored is not None, (
                f"no stored Stage A row for {position['position_id']}"
            )
            decision = bundle.engine.choose_action(state, seed=DECISION_SEED)
            action, record = sample_move(decision, 0.0, None)
            assert action == int(stored["action_id"])
            assert record["sampled"] is False
            assert int(decision.direct_action_id) == int(stored["direct_action_id"])
            assert int(decision.c1_forwards) == int(stored["c1_forwards"])
            assert int(decision.unique_worlds) == int(stored["unique_worlds"])
            assert int(decision.worlds_requested) == int(stored["worlds_requested"])
            if stored["score_margin"] not in ("", None):
                scores = sorted(
                    (candidate.score for candidate in decision.candidates),
                    reverse=True,
                )
                margin = round(float(scores[0] - scores[1]), 6)
                assert margin == pytest.approx(float(stored["score_margin"]), abs=1e-6)
            compared += 1
        assert compared == POSITIONS

    def test_stochastic_engine_class_at_zero_matches_history_too(
        self, models, replayed, stored_rows
    ):
        # Not just the frozen builder: the Phase 16 engine class itself, at
        # tau_r = 0, must reach the same stored decisions through its
        # delegation path.
        from stratego.search.phase16.engine import Phase16StochasticEngine

        frozen = build_stochastic_bundle(
            models, StochasticArm(0.0, 0.0), "TINY", device="cpu"
        )
        engine = Phase16StochasticEngine(
            frozen.engine.model,
            frozen.engine.provider,
            frozen.config,
            rollout_temperature=0.0,
        )
        for position, state, _plan in replayed[:2]:
            stored = stored_rows[(position["position_id"], "p24_b24")]
            decision = engine.choose_action(state, seed=DECISION_SEED)
            assert int(decision.selected_action_id) == int(stored["action_id"])
            assert decision.search_version == "phase12_root_world_search_v1"


class TestNonzeroTemperaturesReproducible:
    def test_sampled_rollouts_reproduce_from_the_seed(self, models, replayed):
        bundle = build_stochastic_bundle(
            models, StochasticArm(0.0, 1.0), "TINY", device="cpu"
        )
        _position, state, _plan = replayed[0]
        first = bundle.engine.choose_action(state, seed=DECISION_SEED, rollout_seed=11)
        again = bundle.engine.choose_action(state, seed=DECISION_SEED, rollout_seed=11)
        assert first.selected_action_id == again.selected_action_id
        assert [c.q_value for c in first.candidates] == [
            c.q_value for c in again.candidates
        ]
        assert first.world_weights == again.world_weights

    def test_sampled_moves_reproduce_from_the_seed(self, models, replayed):
        import numpy as np

        bundle = build_stochastic_bundle(
            models, StochasticArm(0.0, 0.0), "TINY", device="cpu"
        )
        _position, state, _plan = replayed[0]
        decision = bundle.engine.choose_action(state, seed=DECISION_SEED)
        first = sample_move(decision, 0.6, np.random.Generator(np.random.PCG64(21)))
        again = sample_move(decision, 0.6, np.random.Generator(np.random.PCG64(21)))
        assert first[0] == again[0]


class TestSeatEquality:
    def test_control_seat_replays_the_accepted_search_seat(self, models, replayed):
        """StochasticSeat(tau=0, tau_r=0) must equal the accepted SearchSeat
        decision for decision, through the same spec/plan machinery."""
        from stratego.engine.legal_moves import legal_actions
        from stratego.search.phase15.contract import pairing as pairing_of
        from stratego.search.phase15.matchplay import (
            SearchSeat,
            build_owners,
            build_spec,
            opponent_seat,
        )
        from stratego.search.phase15.systems import build_engine
        from stratego.search.phase16.stochastic import StochasticSeat

        owners = build_owners(models, device="cpu")
        arm = StochasticArm(0.0, 0.0)
        bundle = build_stochastic_bundle(models, arm, "TINY", device="cpu")
        accepted_bundle = build_engine("p24_b24", models, "TINY", device="cpu")
        compared = 0
        for _position, state, plan in replayed[:2]:
            if plan.opponent not in owners:
                continue
            reference, _policy = opponent_seat(plan, owners)
            spec = build_spec(plan, reference)
            legal = legal_actions(state)
            accepted_seat = SearchSeat(
                pairing_of("p24_b24"), accepted_bundle.engine, owners=owners
            )
            stochastic_seat = StochasticSeat(arm, bundle, owners=owners)
            accepted_action, accepted_record = accepted_seat.decide(
                state, legal, spec, plan
            )
            stochastic_action, stochastic_record = stochastic_seat.decide(
                state, legal, spec, plan
            )
            assert stochastic_action == accepted_action
            assert (
                stochastic_record["direct_action_id"]
                == accepted_record["direct_action_id"]
            )
            assert stochastic_record["c1_forwards"] == accepted_record["c1_forwards"]
            assert stochastic_record["fallback"] is None
            assert accepted_record["fallback"] is None
            compared += 1
        assert compared > 0
