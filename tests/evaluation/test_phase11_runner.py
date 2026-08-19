"""Phase 11 Agent 2: the validation-run wiring and the privileged pass."""

import json
from pathlib import Path

import pytest

from stratego.evaluation import phase11_runner as runner
from stratego.evaluation.match_spec import EVALUATION_RULES
from stratego.evaluation.policy import PolicyRequirements
from stratego.training.phase11_contract import (
    EVAL_MOVE_BEHAVIOR,
    STRATUM_BINDINGS,
    STRATUM_PHASE8_ANCHOR,
    STRATUM_PHASE9,
)
from stratego.training.phase11_seed import OPPONENT_STRATA, game_match_seed

BANK_PATH = Path("reports/phase_11_data/agent_01_validation_bank.json")


@pytest.fixture(scope="module")
def bank():
    if not BANK_PATH.exists():  # pragma: no cover - Agent 1 artifact required
        pytest.skip("the Agent 1 validation bank artifact is not present")
    return json.loads(BANK_PATH.read_text())


def test_every_stratum_has_an_opponent_binding():
    bound = set(runner.STRATUM_POLICY_IDS) | {STRATUM_PHASE9, STRATUM_PHASE8_ANCHOR}
    assert bound == set(OPPONENT_STRATA)
    for entry in STRATUM_BINDINGS:
        if entry["opponent_policy_id"] is None:
            continue
        assert runner.STRATUM_POLICY_IDS[entry["stratum"]] == entry["opponent_policy_id"]


def test_the_registry_can_build_every_rule_opponent():
    from stratego.evaluation.registry import build_policy

    for policy_id in runner.STRATUM_POLICY_IDS.values():
        assert build_policy(policy_id).ref.policy_id == policy_id


def test_the_observer_seat_is_distinct_from_the_phase9_opponent_seat():
    """Otherwise the self-play stratum would resolve one object for both."""
    observer = runner.observer_ref()
    opponent = runner.neural_opponent_ref(STRATUM_PHASE9)
    assert observer.token != opponent.token
    assert runner.neural_opponent_ref(STRATUM_PHASE8_ANCHOR).token != opponent.token


def test_the_observer_declares_exactly_the_products_it_reads():
    assert Phase11Requirements() == PolicyRequirements(
        observation=True, legal_action_mask=True, public_view=True
    )


def Phase11Requirements():
    return runner.Phase11ObserverPolicy.requirements


def test_the_plan_resolves_setups_by_colour(bank):
    case = bank["cases"][0]
    red_game = runner.game_plan(case, 0)
    blue_game = runner.game_plan(case, 1)
    assert red_game.observer_color == "red"
    assert blue_game.observer_color == "blue"
    assert red_game.red_setup == tuple(case["games"]["0"]["observer"]["setup"])
    assert red_game.blue_setup == tuple(case["games"]["0"]["opponent"]["setup"])
    assert blue_game.blue_setup == tuple(case["games"]["1"]["observer"]["setup"])
    assert blue_game.red_setup == tuple(case["games"]["1"]["opponent"]["setup"])


def test_the_plan_rejects_an_unknown_game_index(bank):
    with pytest.raises(runner.Phase11RunError):
        runner.game_plan(bank["cases"][0], 2)


def test_the_spec_root_seed_is_the_frozen_match_seed(bank):
    case = bank["cases"][0]
    for game_index in (0, 1):
        plan = runner.game_plan(case, game_index)
        spec = runner.build_spec(plan, runner.neural_opponent_ref(STRATUM_PHASE9))
        assert spec.root_seed == plan.match_seed == game_match_seed(plan.game_id)
        assert spec.replicate == game_index
        assert spec.suite_version == runner.PHASE11_RUN_VERSION
        assert spec.rules == EVALUATION_RULES


def test_the_single_game_bank_carries_the_frozen_position(bank):
    plan = runner.game_plan(bank["cases"][3], 0)
    spec = runner.build_spec(plan, runner.neural_opponent_ref(STRATUM_PHASE9))
    setup_bank = runner.single_game_bank(spec, plan)
    pair = setup_bank.pair(spec.setup_pair_id)
    assert pair.red_setup == plan.red_setup
    assert pair.blue_setup == plan.blue_setup


def test_the_eval_move_behaviour_is_the_frozen_one():
    assert EVAL_MOVE_BEHAVIOR["decision_mode"] == "greedy"
    assert EVAL_MOVE_BEHAVIOR["dtype"] == "float32"
    assert EVAL_MOVE_BEHAVIOR["batch_policy"] == "single_request"
    assert EVAL_MOVE_BEHAVIOR["search"] == "none"
    assert runner.observer_ref().policy_version.endswith("float32")


def test_the_bank_covers_every_stratum_and_both_sources(bank):
    cells = {}
    for case in bank["cases"]:
        cells[(case["stratum"], case["setup_source"])] = (
            cells.get((case["stratum"], case["setup_source"]), 0) + 1
        )
    assert len(cells) == len(OPPONENT_STRATA) * 2
    assert set(cells.values()) == {32}
