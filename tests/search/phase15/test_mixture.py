"""The belief-mixture pilot: its algebra, its endpoints and its decision rules."""

import numpy as np
import pytest

from stratego.search.phase12.providers import RemainingCountBeliefProvider
from stratego.search.phase15.gate import _public_state
from stratego.search.phase15.mixture import (
    MIXTURE_LAMBDAS,
    MIXTURE_REFERENCE_PRESET,
    MIXTURE_STAGE1_PRESET,
    MixtureBeliefProvider,
    MixturePairing,
    Phase15MixtureError,
    build_mixture_bundle,
    check_configuration_invariants,
    lambda_token,
    mixture_arm_id,
    mixture_pairing,
    mixture_provider_id,
)
from stratego.search.phase15.mixture_pilot import (
    B24_LARGE_ARM,
    check_reference_arms_reproduce,
    decide_stage2,
    endpoint_arms,
    interior_arms,
    mix_arm_name,
    paired_regret_delta,
    select_lambda,
)
from stratego.search.phase15.providers import build_specialist_provider


# -- the grid ---------------------------------------------------------------


def test_the_instructed_five_weights_and_nothing_else():
    assert MIXTURE_LAMBDAS == (0.00, 0.25, 0.50, 0.75, 1.00)
    assert MIXTURE_STAGE1_PRESET == "LARGE"
    assert MIXTURE_REFERENCE_PRESET == "MEDIUM"


def test_lambda_tokens_are_sortable_and_refuse_a_finer_grid():
    assert [lambda_token(value) for value in MIXTURE_LAMBDAS] == [
        "l000",
        "l025",
        "l050",
        "l075",
        "l100",
    ]
    assert mixture_provider_id(0.5) == "b24_count_mix_l050"
    assert mixture_arm_id(0.5) == "p24_mix_l050"
    with pytest.raises(Phase15MixtureError):
        lambda_token(0.125)
    with pytest.raises(Phase15MixtureError):
        lambda_token(1.5)


def test_the_budgets_are_the_frozen_ones():
    report = check_configuration_invariants()
    assert report["passed"], report["findings"]
    assert report["presets"]["LARGE"]["worlds"] == 64
    assert report["presets"]["MEDIUM"]["worlds"] == 32
    for entry in report["presets"].values():
        assert entry["max_root_candidates"] == 8
        assert entry["beta"] == 0.1


# -- the pairing is duck-typed, not registered -------------------------------


def test_the_frozen_pairing_table_never_learns_a_mixture_name():
    from stratego.search.phase15.contract import (
        ALL_PROVIDERS,
        PAIRINGS_BY_ID,
        PRODUCTION_PAIRING_IDS,
        Phase15SearchError,
        Pairing,
    )

    assert mixture_provider_id(0.5) not in ALL_PROVIDERS
    assert mixture_arm_id(0.5) not in PAIRINGS_BY_ID
    assert mixture_arm_id(0.5) not in PRODUCTION_PAIRING_IDS
    # And the frozen type would have refused it, which is why this pilot
    # carries its own.
    with pytest.raises(Phase15SearchError):
        Pairing(
            pairing_id=mixture_arm_id(0.5),
            move_model="p24",
            provider=mixture_provider_id(0.5),
            kind="search",
            description="",
        )


def test_the_mixture_pairing_carries_what_a_seat_reads():
    target = mixture_pairing(0.25)
    assert isinstance(target, MixturePairing)
    assert (target.pairing_id, target.move_model, target.kind) == (
        "p24_mix_l025",
        "p24",
        "search",
    )
    assert target.provider == mixture_provider_id(0.25)
    assert target.describe()["lambda"] == 0.25


# -- the algebra ------------------------------------------------------------


@pytest.fixture(scope="module")
def components(fake_models):
    learned = build_specialist_provider(
        "b24", fake_models.specialists["b24"], fake_models.move_models
    )
    return learned, RemainingCountBeliefProvider()


def test_the_mixture_is_the_stated_formula(components, midgame_state):
    learned, count = components
    public = _public_state(midgame_state)
    left = learned.predict_marginals(public)
    right = count.predict_marginals(public)
    assert set(left) == set(right) and left
    for lam in MIXTURE_LAMBDAS:
        mixed = MixtureBeliefProvider(learned, count, lam=lam).predict_marginals(public)
        assert set(mixed) == set(left)
        for slot, row in mixed.items():
            expected = lam * np.asarray(left[slot]) + (1.0 - lam) * np.asarray(
                right[slot]
            )
            expected = expected / expected.sum()
            assert np.allclose(row, expected, rtol=0.0, atol=1e-15)
            assert abs(float(row.sum()) - 1.0) <= 1e-12
            assert (row >= 0.0).all()


def test_the_endpoints_are_their_components_to_within_one_ulp(
    components, midgame_state
):
    """The instructed normalization is applied at every lambda, endpoints too.

    A float64 softmax row sums to 1.0 to within a couple of ulps but not
    exactly, so dividing by that sum moves the last bits. That is the honest
    cost of doing what the brief asks. The bound is stated in ulps rather
    than as a decimal because that is what it actually is: measured across
    six random initializations the worst deviation was 2.4 eps, and the
    tolerance here is a few times that so the test does not depend on which
    seed the random-order plugin happened to pick.

    What matters is that nothing downstream can see it, which
    :func:`test_lambda_one_samples_exactly_the_learned_providers_worlds`
    and :func:`test_lambda_one_reproduces_the_frozen_arm_decision` check
    where it counts — on the sampled worlds and on the decision itself.
    """
    tolerance = 8 * float(np.finfo(np.float64).eps)
    learned, count = components
    public = _public_state(midgame_state)
    at_one = MixtureBeliefProvider(learned, count, lam=1.0).predict_marginals(public)
    at_zero = MixtureBeliefProvider(learned, count, lam=0.0).predict_marginals(public)
    for slot, row in learned.predict_marginals(public).items():
        assert np.allclose(at_one[slot], np.asarray(row), rtol=tolerance, atol=0.0)
    for slot, row in count.predict_marginals(public).items():
        assert np.allclose(at_zero[slot], np.asarray(row), rtol=tolerance, atol=0.0)


def test_lambda_one_samples_exactly_the_learned_providers_worlds(
    components, midgame_state
):
    """The ordinal walk is B24's walk, so the endpoint is B24's worlds."""
    learned, count = components
    public = _public_state(midgame_state)
    mixture = MixtureBeliefProvider(learned, count, lam=1.0)
    assert mixture.sample_assignments(public, 16, 4242) == learned.sample_assignments(
        public, 16, 4242
    )


def test_a_mixture_reads_public_state_and_only_public_state(components):
    learned, count = components
    mixture = MixtureBeliefProvider(learned, count, lam=0.5)
    assert mixture.uses_hidden_truth is False
    assert mixture.describe()["uses_hidden_truth"] is False
    assert not hasattr(mixture, "sample_assignments_privileged")
    with pytest.raises(Phase15MixtureError):
        mixture.predict_marginals({"observer_color": "red"})


def test_a_mixture_refuses_a_hidden_truth_component(components):
    from stratego.search.phase12.providers import OracleBeliefProvider

    learned, count = components
    with pytest.raises(Phase15MixtureError):
        MixtureBeliefProvider(
            OracleBeliefProvider(offline_diagnostic=True), count, lam=0.5
        )
    with pytest.raises(Phase15MixtureError):
        MixtureBeliefProvider(learned, count, lam=1.5)


# -- the assembled system ---------------------------------------------------


def test_a_mixture_bundle_is_a_production_engine_over_the_frozen_p24(fake_models):
    bundle = build_mixture_bundle(fake_models, 0.5, "LARGE")
    assert bundle.config.production is True
    assert bundle.config.worlds == 64
    assert bundle.engine.provider.provider_id == mixture_provider_id(0.5)
    assert bundle.pairing.move_model == "p24"
    described = bundle.describe()
    assert described["pairing"]["lambda"] == 0.5
    assert described["roles"]["policy"] == "p24"


def test_the_accepted_match_seat_accepts_a_mixture_arm(fake_models):
    """The duck-typed pairing satisfies the frozen seat, including its check
    that the engine actually carries the provider the arm names."""
    from stratego.search.phase15.matchplay import Phase15MatchError
    from stratego.search.phase15.systems import build_seat

    bundle = build_mixture_bundle(fake_models, 0.75, "LARGE")
    seat = build_seat(bundle, owners={})
    assert seat.kind == "search"
    assert seat.arm_id == mixture_arm_id(0.75)
    assert seat.engine.provider.provider_id == bundle.pairing.provider

    crossed = build_mixture_bundle(fake_models, 0.25, "LARGE")
    crossed.pairing = mixture_pairing(0.75)
    with pytest.raises(Phase15MatchError):
        build_seat(crossed, owners={})


def test_a_mixture_decision_is_legal_and_seed_deterministic(fake_models, midgame_state):
    from stratego.engine.legal_moves import legal_actions

    bundle = build_mixture_bundle(fake_models, 0.5, "MEDIUM")
    legal = set(legal_actions(midgame_state))
    first = bundle.engine.choose_action(midgame_state, seed=606)
    again = bundle.engine.choose_action(midgame_state, seed=606)
    assert first.selected_action_id in legal
    assert first.selected_action_id == again.selected_action_id
    assert first.world_weights == again.world_weights
    assert any(candidate.is_direct for candidate in first.candidates)


def test_lambda_one_reproduces_the_frozen_arm_decision(fake_models, midgame_state):
    from stratego.search.phase15.systems import build_engine

    frozen = build_engine("p24_b24", fake_models, "MEDIUM")
    mixture = build_mixture_bundle(fake_models, 1.0, "MEDIUM")
    left = frozen.engine.choose_action(midgame_state, seed=20260824)
    right = mixture.engine.choose_action(midgame_state, seed=20260824)
    assert left.selected_action_id == right.selected_action_id
    assert left.world_weights == right.world_weights
    assert [round(c.q_value, 12) for c in left.candidates] == [
        round(c.q_value, 12) for c in right.candidates
    ]


# -- reading Stage 1 --------------------------------------------------------


def test_the_sweep_endpoints_and_interior_are_named_correctly():
    assert endpoint_arms() == ("mix_l000_LARGE", "mix_l100_LARGE")
    assert interior_arms() == ["mix_l025_LARGE", "mix_l050_LARGE", "mix_l075_LARGE"]
    assert mix_arm_name(0.5) == "mix_l050_LARGE"


def _regret_rows(arm: str, values):
    return [
        {"arm": arm, "position_id": f"p{index}", "oracle_q_regret": value}
        for index, value in enumerate(values)
    ]


def test_paired_regret_pairs_by_position_and_negative_is_better():
    rows = _regret_rows("a", [0.1, 0.2, 0.3]) + _regret_rows("b", [0.2, 0.3, 0.4])
    report = paired_regret_delta(rows, "a", "b")
    assert report["positions"] == 3
    assert report["delta"] == pytest.approx(-0.1)
    assert report["better_positions"] == 3
    assert report["standard_error"] == pytest.approx(0.0, abs=1e-9)


def test_a_mixture_is_selected_only_when_it_beats_both_endpoints():
    """The interior arm is better than both ends, by more than the noise."""
    rows = (
        _regret_rows("mix_l000_LARGE", [0.30, 0.31, 0.29, 0.30])
        + _regret_rows("mix_l025_LARGE", [0.29, 0.30, 0.28, 0.29])
        + _regret_rows("mix_l050_LARGE", [0.10, 0.11, 0.09, 0.10])
        + _regret_rows("mix_l075_LARGE", [0.20, 0.21, 0.19, 0.20])
        + _regret_rows("mix_l100_LARGE", [0.32, 0.33, 0.31, 0.32])
    )
    summary = {
        arm: {"oracle_q_regret_mean": float(np.mean(values))}
        for arm, values in (
            ("mix_l025_LARGE", [0.29, 0.30, 0.28, 0.29]),
            ("mix_l050_LARGE", [0.10, 0.11, 0.09, 0.10]),
            ("mix_l075_LARGE", [0.20, 0.21, 0.19, 0.20]),
        )
    }
    report = select_lambda(rows, summary)
    assert report["selected_arm"] == "mix_l050_LARGE"
    assert report["selected_lambda"] == 0.50
    assert report["stage2_authorized"] is True


def test_no_mixture_is_selected_when_the_endpoints_are_not_beaten():
    """A monotone sweep has no interior winner, and the rule must say so."""
    rows = (
        _regret_rows("mix_l000_LARGE", [0.40, 0.41, 0.39, 0.40])
        + _regret_rows("mix_l025_LARGE", [0.35, 0.36, 0.34, 0.35])
        + _regret_rows("mix_l050_LARGE", [0.30, 0.31, 0.29, 0.30])
        + _regret_rows("mix_l075_LARGE", [0.25, 0.26, 0.24, 0.25])
        + _regret_rows("mix_l100_LARGE", [0.20, 0.21, 0.19, 0.20])
    )
    summary = {
        "mix_l025_LARGE": {"oracle_q_regret_mean": 0.35},
        "mix_l050_LARGE": {"oracle_q_regret_mean": 0.30},
        "mix_l075_LARGE": {"oracle_q_regret_mean": 0.25},
    }
    report = select_lambda(rows, summary)
    assert report["selected_arm"] is None
    assert report["stage2_authorized"] is False
    assert report["findings"]


def test_a_tie_inside_the_standard_error_is_not_a_winner():
    rows = (
        _regret_rows("mix_l000_LARGE", [0.30, 0.10, 0.50, 0.20])
        + _regret_rows("mix_l050_LARGE", [0.29, 0.11, 0.49, 0.21])
        + _regret_rows("mix_l100_LARGE", [0.31, 0.09, 0.51, 0.19])
    )
    summary = {"mix_l050_LARGE": {"oracle_q_regret_mean": 0.275}}
    report = select_lambda(rows, summary, lambdas=(0.0, 0.5, 1.0))
    assert report["selected_arm"] is None


# -- reading Stage 2 --------------------------------------------------------


def _rung(ewr, *, fallbacks=0, delta=None):
    return {
        "ewr": ewr,
        "fallbacks": fallbacks,
        "fallback_reasons": {},
        "paired_vs_reference": None if delta is None else {"delta": delta},
    }


def test_the_mixture_is_adopted_only_when_it_recovers_half_the_regression():
    rungs = {
        "medium": _rung(0.933),
        "large": _rung(0.858),
        "mix": _rung(0.910, delta=-0.023),
    }
    report = decide_stage2(rungs, medium_arm="medium", large_arm="large", mix_arm="mix")
    assert report["fraction_of_regression_recovered"] > 0.5
    assert report["adopt_mixture_for_deeper_search"] is True


def test_a_mixture_that_recovers_little_keeps_medium():
    rungs = {
        "medium": _rung(0.933),
        "large": _rung(0.858),
        "mix": _rung(0.870, delta=-0.063),
    }
    report = decide_stage2(rungs, medium_arm="medium", large_arm="large", mix_arm="mix")
    assert report["fraction_of_regression_recovered"] < 0.5
    assert report["adopt_mixture_for_deeper_search"] is False
    assert report["recommendation"] == "keep MEDIUM + B24"


def test_a_fallback_disqualifies_a_mixture_that_would_otherwise_pass():
    rungs = {
        "medium": _rung(0.933),
        "large": _rung(0.858),
        "mix": _rung(0.930, fallbacks=3, delta=-0.003),
    }
    rungs["mix"]["fallback_reasons"] = {"search_error": 3}
    report = decide_stage2(rungs, medium_arm="medium", large_arm="large", mix_arm="mix")
    assert report["correctness_clean"] is False
    assert report["adopt_mixture_for_deeper_search"] is False


# -- the Stage 2 reproduction check -----------------------------------------


def _game_row(arm, preset, board, *, outcome="win", score=1.0, plies=400, actions=(1, 2, 3)):
    return {
        "arm_id": arm,
        "preset_id": preset,
        "board_id": board,
        "outcome": outcome,
        "effective_score": score,
        "plies": plies,
        "actions": list(actions),
    }


def test_a_reference_arm_that_reproduces_its_stored_row_passes():
    stored = [_game_row("p24_b24", "LARGE", "b0"), _game_row("p24_b24", "MEDIUM", "b0")]
    report = check_reference_arms_reproduce(list(stored), stored)
    assert report["passed"] is True
    assert report["games_compared"] == 2


def test_a_reference_arm_that_drifts_is_reported_not_swallowed():
    stored = [_game_row("p24_b24", "LARGE", "b0")]
    fresh = [_game_row("p24_b24", "LARGE", "b0", outcome="loss", score=0.0)]
    report = check_reference_arms_reproduce(fresh, stored)
    assert report["passed"] is False
    assert any("outcome" in finding for finding in report["findings"])

    diverged = [_game_row("p24_b24", "LARGE", "b0", actions=(1, 2, 9))]
    assert check_reference_arms_reproduce(diverged, stored)["passed"] is False


def test_no_overlap_is_a_failure_rather_than_a_vacuous_pass():
    stored = [_game_row("p24_b24", "LARGE", "b0")]
    fresh = [_game_row("p24_mix_l050", "LARGE", "b0")]
    report = check_reference_arms_reproduce(fresh, stored)
    assert report["games_compared"] == 0
    assert report["passed"] is False


# -- the write-up -----------------------------------------------------------


def test_the_report_renders_from_the_real_artifacts_when_they_exist(repository_root):
    """A negative Stage 1 must still produce a complete document."""
    import json

    from stratego.search.phase15.mixture_report_text import build_mixture_report

    root = repository_root / "reports" / "phase15"
    gate_path = root / "agent_02_mixture_gate.json"
    stage1_path = root / "agent_02_mixture_stage1.json"
    if not (gate_path.is_file() and stage1_path.is_file()):
        pytest.skip("the mixture pilot has not been run in this tree")
    text = build_mixture_report(
        gate=json.loads(gate_path.read_text()),
        stage1=json.loads(stage1_path.read_text()),
        stage2=None,
    )
    assert "Belief-mixture pilot" in text
    assert "Stage 1 — the position diagnostic" in text
    # Every arm of the sweep is named somewhere in the document.
    for lam in MIXTURE_LAMBDAS:
        assert f"lambda={lam:.2f}" in text
