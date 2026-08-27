"""Match seats, the diagnostic positions, and the parallel execution layer."""

import pytest

from stratego.search.phase15.boards import board_plan
from stratego.search.phase15.contract import pairing as pairing_of
from stratego.search.phase15.execution import Task
from stratego.search.phase15.matchplay import (
    DirectSeat,
    Phase15MatchError,
    SearchSeat,
    SeatProbe,
    build_owners,
    build_spec,
    opponent_seat,
    outcome_of,
    play_board,
    single_game_bank,
)
from stratego.search.phase15.systems import build_engine


@pytest.fixture(scope="module")
def owners(models):
    return build_owners(models, device="cpu")


@pytest.fixture(scope="module")
def plan(setup_sources):
    return board_plan("stress_berserker", "targeted_family", "red", 0, setup_sources)


# -- spec and bank ----------------------------------------------------------


def test_the_bank_carries_the_oriented_setups(plan):
    reference, _policy = opponent_seat(plan, {})
    spec = build_spec(plan, reference)
    bank = single_game_bank(spec, plan)
    red, blue = spec.resolve_setups(bank)
    assert tuple(red) == plan.red_setup
    assert tuple(blue) == plan.blue_setup


def test_a_neural_opponent_without_its_owner_is_refused(setup_sources):
    neural = board_plan("phase9_anchor", "neutral_v1", "red", 0, setup_sources)
    with pytest.raises(Phase15MatchError, match="needs a phase9_anchor owner"):
        opponent_seat(neural, {})


def test_every_stress_opponent_resolves_to_an_accepted_policy(setup_sources):
    from stratego.search.phase15.contract import RULE_OPPONENT_POLICY_IDS

    for opponent, policy_id in RULE_OPPONENT_POLICY_IDS.items():
        board = board_plan(opponent, "neutral_v1", "red", 0, setup_sources)
        reference, policy = opponent_seat(board, {})
        assert policy_id in reference.policy_id


def test_outcome_thresholds():
    assert outcome_of(1.0) == "win"
    assert outcome_of(0.5) == "draw"
    assert outcome_of(0.0) == "loss"


# -- seats ------------------------------------------------------------------


def test_a_direct_seat_needs_its_own_move_model(models, owners):
    seat = DirectSeat(pairing_of("p24_direct"), owners)
    assert seat.kind == "direct"
    with pytest.raises(Phase15MatchError, match="is not a direct pairing"):
        DirectSeat(pairing_of("p24_b18"), owners)


def test_a_search_seat_refuses_a_mismatched_engine(models, owners):
    bundle = build_engine("p18_b18", models, "TINY")
    with pytest.raises(Phase15MatchError, match="wants provider"):
        SearchSeat(pairing_of("p18_b24"), bundle.engine, owners=owners)


def test_a_search_seat_falls_back_to_its_own_direct_model(models, owners, plan, midgame_state):
    from stratego.engine.legal_moves import legal_actions

    bundle = build_engine("p24_b18", models, "TINY")
    seat = SearchSeat(pairing_of("p24_b18"), bundle.engine, owners=owners, time_cap=1e-9)
    reference, _policy = opponent_seat(plan, owners)
    spec = build_spec(plan, reference)
    legal = legal_actions(midgame_state)
    action, record = seat.decide(midgame_state, legal, spec, plan)
    assert record["fallback"] == "timeout"
    assert action in legal
    assert seat.fallbacks["timeout"] == 1


def test_a_search_seat_records_its_diagnostics(models, owners, plan, midgame_state):
    from stratego.engine.legal_moves import legal_actions

    bundle = build_engine("p18_b24", models, "TINY")
    seat = SearchSeat(pairing_of("p18_b24"), bundle.engine, owners=owners)
    reference, _policy = opponent_seat(plan, owners)
    spec = build_spec(plan, reference)
    action, record = seat.decide(midgame_state, legal_actions(midgame_state), spec, plan)
    assert record["fallback"] is None
    assert record["unique_worlds"] >= 1
    assert record["c1_forwards"] > 1
    assert record["direct_action_id"] is not None
    assert record["score_margin"] is None or isinstance(record["score_margin"], float)


# -- one whole game ---------------------------------------------------------


def test_a_direct_game_completes_and_records_its_board(models, owners, plan):
    bundle = build_engine("p18_direct", models, "TINY")
    seat = DirectSeat(bundle.pairing, owners)
    record = play_board(plan, seat, owners, preset_id="direct")
    row = record.row()
    assert row["board_id"] == plan.board_id
    assert row["arm_id"] == "p18_direct"
    assert row["opponent"] == "stress_berserker"
    assert row["outcome"] in ("win", "draw", "loss")
    assert row["player_decisions"] > 0
    assert row["move_change_rate"] is None
    assert row["fallbacks"] == 0


def test_two_plays_of_one_board_agree_exactly(models, owners, plan):
    bundle = build_engine("p24_b18", models, "TINY")
    first = play_board(plan, SearchSeat(bundle.pairing, bundle.engine, owners=owners), owners)
    second = play_board(plan, SearchSeat(bundle.pairing, bundle.engine, owners=owners), owners)
    assert first.effective_score == second.effective_score
    assert first.plies == second.plies
    assert first.move_changes == second.move_changes


def test_the_probe_fires_and_reports(models, owners, plan):
    bundle = build_engine("p18_b18", models, "TINY")
    seat = SearchSeat(bundle.pairing, bundle.engine, owners=owners)
    probe = SeatProbe(reference=seat.direct_policy, interval=8, budget=3)
    play_board(plan, seat, owners, probe=probe)
    report = probe.summary()
    assert report["permutation_checks"] > 0
    assert report["direct_agreement_checks"] > 0
    assert report["passed"] is True


def test_the_probe_never_spends_its_budget_on_the_opening():
    probe = SeatProbe(interval=4, budget=2)
    assert probe.due(0) is False
    assert probe.due(4) is True


# -- positions --------------------------------------------------------------


def test_positions_are_eligible_and_replay_exactly(models, owners, setup_sources):
    from stratego.search.phase15.positions import (
        MIN_PLY,
        MIN_UNRESOLVED,
        build_manifest,
        materialize_positions,
        play_for_positions,
    )

    board = board_plan("tactical_rule_based", "neutral_v1", "blue", 300, setup_sources)
    found = play_for_positions(board, "p18", owners, per_game=3)
    assert found
    for position in found:
        assert position.ply >= MIN_PLY
        assert position.unresolved >= MIN_UNRESOLVED
    manifest = build_manifest(found, generated_utc="2026-01-01T00:00:00Z")
    replayed = materialize_positions(manifest, sources=setup_sources, verify=True)
    assert len(replayed) == len(found)


def test_a_tampered_position_is_refused(models, owners, setup_sources):
    from stratego.search.phase15.positions import (
        Phase15PositionError,
        build_manifest,
        materialize_positions,
        play_for_positions,
    )

    board = board_plan("strategic_rule_based", "neutral_v1", "red", 301, setup_sources)
    found = play_for_positions(board, "p24", owners, per_game=2)
    manifest = build_manifest(found, generated_utc="2026-01-01T00:00:00Z")
    manifest["positions"][0]["observation_sha256"] = "f" * 64
    with pytest.raises(Phase15PositionError, match="replayed observation hashes"):
        materialize_positions(manifest, sources=setup_sources, verify=True)


def test_position_cells_balance_the_two_observers():
    from stratego.search.phase15.positions import position_cells

    cells = position_cells(10)
    assert len(cells) == 20
    assert sum(1 for cell in cells if cell[0] == "p18") == 10
    assert len({cell[4] for cell in cells}) == 10


def test_evenly_spaced_keeps_both_endpoints():
    from stratego.search.phase15.positions import evenly_spaced

    assert evenly_spaced([1, 2, 3, 4, 5], 3) == [1, 3, 5]
    assert evenly_spaced([1, 2], 5) == [1, 2]
    assert evenly_spaced([], 3) == []


# -- execution --------------------------------------------------------------


def test_the_serial_path_reproduces_a_direct_play(repository_root, plan, models, owners):
    from stratego.search.phase15.execution import run_pack

    task = Task(arm_id="p18_direct", preset_name="TINY", board_id=plan.board_id)
    results = run_pack(
        [task], root=str(repository_root), workers=1, with_anchor=True
    )
    bundle = build_engine("p18_direct", models, "TINY")
    reference = play_board(plan, DirectSeat(bundle.pairing, owners), owners)
    assert results[0]["row"]["effective_score"] == reference.effective_score
    assert results[0]["row"]["plies"] == reference.plies


def test_an_empty_pack_returns_nothing(repository_root):
    from stratego.search.phase15.execution import run_pack

    assert run_pack([], root=str(repository_root)) == []


def test_a_task_key_is_its_identity():
    task = Task(arm_id="p18_b18", preset_name="TINY", board_id="b")
    assert task.key == ("p18_b18", "TINY", "b")


def test_the_serial_path_leaves_the_process_untouched(repository_root, plan):
    """A reference run must not mutate global state its caller shares.

    An earlier version pinned `OMP_NUM_THREADS` inside the worker
    initializer. On the in-process serial path that leaked into the caller's
    environment and broke the accepted worker-pool tests two files away —
    a failure with no visible connection to its cause. Both the environment
    and torch's thread count must come back unchanged.
    """
    import os

    import torch

    from stratego.search.phase15.execution import run_pack

    before_threads = torch.get_num_threads()
    before_env = dict(os.environ)

    run_pack(
        [Task(arm_id="p18_direct", preset_name="TINY", board_id=plan.board_id)],
        root=str(repository_root),
        workers=1,
        with_anchor=True,
    )

    assert torch.get_num_threads() == before_threads
    assert dict(os.environ) == before_env
