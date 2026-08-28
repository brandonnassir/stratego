"""Phase 17 Agent 2: the fixed-transition window and the forced rebind."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stratego.engine.constants import BLUE, RED
from stratego.training.phase17.move_contract import (
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_TERMINAL,
    parse_game_id,
)
from stratego.training.phase17.move_snapshot import (
    CurrentMovePolicy,
    reproduce_sample,
    snapshot_from_model,
)
from stratego.training.phase17.move_start import load_phase17_move_weights
from stratego.training.phase17.transition_collector import (
    FixedTransitionCollector,
    Phase17CollectorError,
    ScheduledPhase17Game,
    collector_semantics,
)
from stratego.training.phase17.transition_schema import assert_unique, validate_transition

from .test_move_support import DeterministicSetupProvider, perturbed_copy

RUN = "RUN-TEST-A"


@pytest.fixture(scope="module")
def model():
    return load_phase17_move_weights(device="cpu")["model"]


def build_collector(model, *, population=6, budget=120, offset=0, iteration=1):
    cell = CurrentMovePolicy(snapshot_from_model(model, device="cpu"), iteration=iteration)
    collector = FixedTransitionCollector(
        run_id=RUN,
        cell=cell,
        setup_provider=DeterministicSetupProvider(offset=offset),
        population=population,
        budget=budget,
    )
    return cell, collector


# ---------------------------------------------------------------------------
# Exactly the budget
# ---------------------------------------------------------------------------


def test_a_window_emits_exactly_the_configured_transition_budget(model):
    _cell, collector = build_collector(model, budget=120)
    result = collector.collect_window()
    assert result.transitions_harvested == 120
    assert len(result.rows) == 120
    assert result.summary()["exact_budget"] is True


@pytest.mark.parametrize("budget", [37, 64, 150])
def test_the_budget_is_exact_at_several_sizes(model, budget):
    """The exact count does not depend on how the budget divides the population."""
    _cell, collector = build_collector(model, population=8, budget=budget, offset=5)
    rows = []
    for _ in range(3):
        result = collector.collect_window()
        assert len(result.rows) == budget
        assert result.transitions_harvested == budget
        rows.extend(result.rows)
    assert len(rows) == 3 * budget
    assert assert_unique(rows)["duplicates"] == 0
    assert max(row.bootstrap_age_windows for row in rows) >= 1


def test_the_budget_is_exact_across_a_short_and_long_game_mixture(model):
    """Games end and are replaced mid-window; traces span every window.

    This is also the configuration that exercises the window-close seal: a game
    whose terminating action is the one that reaches the budget is still seated
    when the loop exits, and must close on its real outcome rather than on a
    bootstrap of a terminal position.
    """
    _cell, collector = build_collector(model, population=4, budget=150, offset=0)
    rows = []
    sealed = 0
    for _ in range(4):
        result = collector.collect_window()
        assert len(result.rows) == 150
        sealed += result.sealed_at_boundary
        rows.extend(result.rows)

    assert assert_unique(rows)["duplicates"] == 0
    assert collector.games_completed > 0
    assert max(row.bootstrap_age_windows for row in rows) >= 2
    provenance = {row.target_provenance for row in rows}
    assert provenance == {PROVENANCE_BOOTSTRAP, PROVENANCE_TERMINAL}
    assert sealed > 0, "the window-close seal path was never exercised"


def test_every_transition_is_emitted_exactly_once_across_windows(model):
    _cell, collector = build_collector(model, population=6, budget=100)
    rows = []
    for _ in range(4):
        rows.extend(collector.collect_window().rows)
    assert assert_unique(rows) == {"rows": len(rows), "duplicates": 0}
    assert len(rows) == 400


def test_both_seats_appear_and_every_row_validates(model):
    _cell, collector = build_collector(model, budget=150)
    result = collector.collect_window()
    colors = {row.color for row in result.rows}
    assert colors == {RED, BLUE}
    for row in result.rows:
        validate_transition(row)
        assert row.perspective_player == row.color
        assert parse_game_id(row.game_id)["run_id"] == RUN
        assert row.observation.shape == (127, 10, 10)


def test_the_stored_draw_replays_from_the_stored_distribution_and_seed(model):
    _cell, collector = build_collector(model, budget=80)
    result = collector.collect_window()
    for row in result.rows:
        assert (
            reproduce_sample(
                row.behavior_probabilities, row.legal_actions, seed=row.action_seed
            )
            == row.sampled_action
        )


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


def test_unfinished_traces_are_bootstrapped_and_finished_ones_are_not(model):
    _cell, collector = build_collector(model, population=4, budget=120)
    result = collector.collect_window()
    provenance = {row.target_provenance for row in result.rows}
    assert PROVENANCE_BOOTSTRAP in provenance
    assert result.boundary_rows + result.terminal_rows == len(result.rows)
    for row in result.rows:
        if row.target_provenance == PROVENANCE_TERMINAL:
            assert row.boundary_target_divergence == pytest.approx(0.0)


def test_a_boundary_prediction_is_taken_for_every_open_seat(model):
    """Both players, per the instruction: a seat with pending rows gets a tail."""
    _cell, collector = build_collector(model, population=3, budget=60)
    collector.collect_window()
    runner = collector.active_runners()[0]
    tails = runner.boundary_predictions(collector.cell)
    assert set(tails) <= {RED, BLUE}
    for tail in tails.values():
        assert tail.kind == "bootstrap"
        assert tail.model_state_digest == collector.cell.digest
        assert sum(tail.wdl_prediction) == pytest.approx(1.0, abs=1e-4)


def test_an_unapplied_request_is_dropped_at_the_boundary(model):
    """A pending decision may not be applied under the next window's weights."""
    _cell, collector = build_collector(model, population=8, budget=20)
    result = collector.collect_window()
    assert result.dropped_pending > 0
    assert all(runner.pending is None for runner in collector.active_runners())


def test_bootstrap_age_grows_for_a_trace_that_survives_windows(model):
    _cell, collector = build_collector(model, population=4, budget=90)
    ages = []
    for _ in range(3):
        ages.append({row.bootstrap_age_windows for row in collector.collect_window().rows})
    assert 0 in ages[0]
    assert max(max(entry) for entry in ages) >= 1


def test_divergence_is_recorded_when_a_bootstrapped_game_finishes(model):
    _cell, collector = build_collector(model, population=4, budget=110, offset=3)
    summaries = []
    for _ in range(4):
        summaries.append(collector.collect_window().divergence_summary())
    assert any(entry["bootstrapped_rows"] > 0 for entry in summaries)
    assert any(entry["max_advantage_divergence"] > 0.0 for entry in summaries)
    assert all(entry["is_a_gate"] is False for entry in summaries)


# ---------------------------------------------------------------------------
# The forced in-flight rebind -- Agent 2 instruction section 4
# ---------------------------------------------------------------------------


def test_an_in_flight_game_takes_its_next_decisions_from_the_new_weights(model):
    """The Phase 16 blocker, proven on numbers rather than metadata.

    1. run games under snapshot A;
    2. build snapshot B with deliberately different legal logits;
    3. rebind without ending or recreating any game;
    4. take further Red and Blue decisions in those same games;
    5. the new rows carry B's digest AND B's behavior distribution;
    6. the pre-rebind rows still carry A's digest and A's distribution.
    """
    cell, collector = build_collector(model, population=4, budget=60)
    digest_a = cell.digest
    before = collector.collect_window().rows
    live = {runner.game_id for runner in collector.active_runners()}
    assert live, "the population must still hold games in flight"

    other = perturbed_copy(model, scale=1.25, seed=4)
    report = cell.rebind_from_model(other, iteration=2)
    digest_b = report["model_state_digest_after"]
    assert report["changed"] is True
    assert digest_b != digest_a

    after = collector.collect_window().rows
    continued = [row for row in after if row.game_id in live]
    assert continued, "no in-flight game produced a post-rebind decision"
    assert {row.color for row in continued} == {RED, BLUE}

    # 5: every post-rebind decision is bound to B, and its stored distribution
    # is the one B produces on that exact observation.
    snapshot_b = cell.snapshot
    for row in continued:
        assert row.behavior_model_state_digest == digest_b
    checked = 0
    for row in continued[:12]:
        observation = torch.from_numpy(row.observation[None, ...])
        with torch.no_grad():
            logits_b = snapshot_b.model.forward_observation(observation).policy_logits[0]
            logits_a = _snapshot_of(model).model.forward_observation(observation).policy_logits[0]
        stored = np.asarray(row.behavior_probabilities, dtype=np.float64)
        from_b = _legal_softmax(logits_b, row.legal_actions, row.color)
        from_a = _legal_softmax(logits_a, row.legal_actions, row.color)
        assert np.allclose(stored, from_b, atol=1e-5)
        assert not np.allclose(stored, from_a, atol=1e-3)
        checked += 1
    assert checked > 0

    # 6: the rows stored before the rebind are untouched and still name A.
    assert {row.behavior_model_state_digest for row in before} == {digest_a}
    assert all(row.behavior_snapshot_iteration == 1 for row in before)


def _snapshot_of(model):
    return snapshot_from_model(model, device="cpu")


def _legal_softmax(policy_logits, legal_actions, color):
    """The stored distribution's own definition, recomputed for the check."""
    from stratego.model.action_frame import absolute_action_to_model

    indices = [absolute_action_to_model(int(action), int(color)) for action in legal_actions]
    values = policy_logits[indices].to(torch.float64)
    weights = torch.exp(values - values.max())
    return (weights / weights.sum()).numpy()


def test_a_decision_prepared_under_stale_weights_is_refused(model):
    """The structural guard: `apply_neural` compares object identity, not a token."""
    cell, collector = build_collector(model, population=2, budget=40)
    collector.fill()
    runner = collector.active_runners()[0]
    request = runner.advance()
    assert request is not None
    cell.rebind_from_model(perturbed_copy(model), iteration=2)
    with pytest.raises(Phase17CollectorError, match="stale snapshot"):
        runner.apply_neural(
            torch.zeros(10000, dtype=torch.float32),
            torch.tensor([0.4, 0.3, 0.3], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# The participant ledger and the structural refusals
# ---------------------------------------------------------------------------


def test_the_runtime_ledger_shows_only_the_current_raw_policy(model):
    cell, collector = build_collector(model, population=5, budget=120)
    collector.collect_window()
    cell.rebind_from_model(perturbed_copy(model), iteration=2)
    collector.collect_window()
    ledger = collector.participant_ledger()
    assert ledger["holds"] is True
    assert ledger["unknown_model_states"] == {}
    assert ledger["rule_or_stress_decisions"] == 0
    assert ledger["historical_participants"] == 0
    assert ledger["search_participants"] == 0
    assert ledger["distinct_acting_model_states"] == 2
    assert set(ledger["transitions_by_model_state"]) <= set(cell.known_digests())
    assert ledger["seats"] == {"red": ledger["policy_token"], "blue": ledger["policy_token"]}


def test_the_ledger_catches_a_digest_the_cell_never_held(model):
    _cell, collector = build_collector(model, population=2, budget=30)
    collector.collect_window()
    collector.observed_digests["f" * 64] = 1
    ledger = collector.participant_ledger()
    assert ledger["holds"] is False
    assert ledger["unknown_model_states"] == {"f" * 64: 1}


def test_a_collector_without_a_setup_provider_is_refused(model):
    cell = CurrentMovePolicy(snapshot_from_model(model, device="cpu"), iteration=1)
    with pytest.raises(Phase17CollectorError, match="no silent library fallback"):
        FixedTransitionCollector(
            run_id=RUN, cell=cell, setup_provider=None, population=2
        )


def test_a_collector_needs_a_live_cell(model):
    with pytest.raises(Phase17CollectorError, match="CurrentMovePolicy"):
        FixedTransitionCollector(
            run_id=RUN,
            cell=snapshot_from_model(model, device="cpu"),
            setup_provider=DeterministicSetupProvider(),
            population=2,
        )


def test_the_scheduled_game_makes_both_seats_learners():
    scheduled = ScheduledPhase17Game(game_id="g", setup_root_seed=1)
    assert scheduled.learner_color is None
    assert scheduled.opponent_kind == "current_policy"
    assert scheduled.red_policy_identity == scheduled.blue_policy_identity
    assert scheduled.red_policy_seed is None
    assert scheduled.historical_snapshot_identity is None
    assert set(scheduled.learner_sides) == {"red", "blue"}


def test_a_rule_decision_is_structurally_unreachable(model):
    """`_rule_decision` raises, so stop condition I5 is an exception not a claim."""
    _cell, collector = build_collector(model, population=1, budget=10)
    collector.fill()
    runner = collector.active_runners()[0]
    with pytest.raises(Phase17CollectorError, match="100% current-policy"):
        runner._rule_decision([], None)


# ---------------------------------------------------------------------------
# Carry state
# ---------------------------------------------------------------------------


def test_the_carry_state_round_trips_without_duplicating_or_omitting(model):
    """A resumed collector continues the same traces from the same cursor."""
    _cell, collector = build_collector(model, population=4, budget=80)
    first = collector.collect_window().rows
    state = collector.state()
    assert state["run_id"] == RUN
    assert state["iteration"] == 1
    assert state["carry"], "an open population must carry its traces"

    resumed_cell, resumed = build_collector(model, population=4, budget=80)
    resumed.restore_counters(state)
    assert resumed.restore_seating(state) == len(state["seated"])
    resumed.fill()
    assert [runner.game_id for runner in resumed.slots] == [
        entry["game_id"] for entry in state["seated"]
    ]
    assert resumed.draw_counts == state["draw_counts"]
    restored = resumed.restore_traces(state)
    assert restored == len(state["carry"])
    for entry in state["carry"]:
        trace = resumed.slots[entry["slot"]].traces[entry["color"]]
        assert trace.emitted == entry["emitted"]
        assert trace.pending == 0
        assert trace.windows_spanned == entry["windows_spanned"]
    assert resumed.iteration == collector.iteration
    assert resumed.transitions_emitted == len(first)


def test_carry_state_from_another_run_is_refused(model):
    _cell, collector = build_collector(model, population=2, budget=30)
    collector.collect_window()
    state = collector.state()
    state["run_id"] = "RUN-OTHER"
    with pytest.raises(Phase17CollectorError, match="belongs to run"):
        collector.restore_counters(state)


def test_carry_state_with_a_different_population_is_refused(model):
    _cell, collector = build_collector(model, population=2, budget=30)
    collector.collect_window()
    state = collector.state()
    _other_cell, other = build_collector(model, population=3, budget=30)
    with pytest.raises(Phase17CollectorError, match="slots"):
        other.restore_counters(state)


def test_a_window_can_stop_early_and_says_so(model):
    _cell, collector = build_collector(model, population=3, budget=10_000)
    calls = {"n": 0}

    def should_continue():
        calls["n"] += 1
        return calls["n"] < 4

    result = collector.collect_window(should_continue=should_continue)
    assert result.stopped_early is True
    assert len(result.rows) == result.transitions_harvested < 10_000


def test_collector_semantics_names_the_three_differences():
    semantics = collector_semantics()
    assert semantics["budget"].startswith("exactly")
    assert "no per-runner copy" in semantics["resolution"]
    assert semantics["search"] == "not imported and not reachable"
