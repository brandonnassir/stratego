"""Phase 17 Agent 2: the structural no-search gate and the contract refusals.

Agent 2 instruction section 7: the collector/trainer dependency graph must not
import Phase 12/15/16 search players, belief-world providers, or search
configuration. This is checked by *importing the move modules in a clean
interpreter and reading `sys.modules`*, not by grepping the source: an import
that only happens at call time would pass a grep and fail here.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from stratego.training.phase17.move_contract import (
    FORBIDDEN_TRAINING_PARTICIPANTS,
    MOVE_TRANSITION_VERSION,
    Phase17MoveError,
    assert_current_policy_only,
    contract_digest,
    game_id,
    move_contract_document,
    parse_game_id,
    require_run_id,
)

MOVE_MODULES = (
    "stratego.training.phase17.move_contract",
    "stratego.training.phase17.move_snapshot",
    "stratego.training.phase17.move_start",
    "stratego.training.phase17.move_loss",
    "stratego.training.phase17.move_trainer",
    "stratego.training.phase17.transition_schema",
    "stratego.training.phase17.transition_targets",
    "stratego.training.phase17.transition_collector",
)

#: Substrings that may not appear in the import closure of the move half.
FORBIDDEN_MODULE_MARKERS = (
    "stratego.search",
    "stratego.belief",
    "phase12",
    "phase15",
    "phase16.collector",
    "phase16.population",
    "phase16.runner",
    "phase16.seat",
    "phase16.setups",
)


def _import_closure() -> list:
    """Every `stratego.*` module loaded by importing the move half, cleanly."""
    program = (
        "import json, sys\n"
        f"for name in {list(MOVE_MODULES)!r}:\n"
        "    __import__(name)\n"
        "print(json.dumps(sorted(n for n in sys.modules if n.startswith('stratego'))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def closure():
    return _import_closure()


def test_the_move_half_imports_no_search_or_belief_module(closure):
    offending = [
        name
        for name in closure
        for marker in FORBIDDEN_MODULE_MARKERS
        if marker in name
    ]
    assert offending == [], f"the move half's import closure reaches {offending}"


def test_the_move_half_does_import_the_accepted_components(closure):
    """The reuse claim, checked the same way as the refusal."""
    for name in (
        "stratego.engine.observation",
        "stratego.model.losses",
        "stratego.training.phase9_contract",
        "stratego.training.phase9_loss",
        "stratego.training.phase9_collector",
        "stratego.training.phase16.schedules",
        "stratego.training.phase16.trainer",
    ):
        assert name in closure


def test_no_rule_or_stress_policy_module_is_reachable(closure):
    """A rule seat is not merely unused: its registry is never even loaded."""
    assert "stratego.evaluation.registry" in closure, (
        "the accepted Phase 9 collector legitimately imports the registry; if "
        "this changes, the assertion below is measuring nothing"
    )
    assert not any(name.startswith("stratego.search") for name in closure)


# ---------------------------------------------------------------------------
# The configuration refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(FORBIDDEN_TRAINING_PARTICIPANTS))
def test_every_forbidden_participant_name_is_refused(name):
    with pytest.raises(Phase17MoveError, match="evaluation instruments only"):
        assert_current_policy_only([name])


def test_a_forbidden_name_is_caught_inside_a_compound_identifier():
    for name in (
        "phase16_stress_chaos",
        "opponent:p24",
        "historical-H035",
        "belief_search_medium",
    ):
        with pytest.raises(Phase17MoveError):
            assert_current_policy_only([name])


def test_a_mapping_is_checked_on_both_keys_and_values():
    with pytest.raises(Phase17MoveError):
        assert_current_policy_only({"red": "current_raw", "blue": "phase9_anchor"})
    report = assert_current_policy_only({"red": "current_raw", "blue": "current_raw"})
    assert report["offending"] == 0
    assert report["checked"] == 4


def test_the_current_policy_names_are_accepted():
    report = assert_current_policy_only(
        ["current_raw_move", "P17RAW", "phase17_current_raw_move_v1"]
    )
    assert report["offending"] == 0
    assert assert_current_policy_only(None)["checked"] == 0


# ---------------------------------------------------------------------------
# Identity plumbing
# ---------------------------------------------------------------------------


def test_a_game_id_round_trips_and_refuses_bad_parts():
    identifier = game_id("RUN-2026-A", 12, 345)
    assert parse_game_id(identifier) == {
        "run_id": "RUN-2026-A",
        "slot": 12,
        "draw": 345,
    }
    with pytest.raises(Phase17MoveError, match="slot must be"):
        game_id("RUN-2026-A", -1, 0)
    with pytest.raises(Phase17MoveError, match="draw must be"):
        game_id("RUN-2026-A", 0, 10_000_000)
    with pytest.raises(Phase17MoveError, match="not a Phase 17 game id"):
        parse_game_id("phase16_game_v1|ms=1|arm=a|slot=0000|draw=000000")


def test_a_run_id_with_a_separator_is_refused():
    for bad in ("RUN|A", "RUN:A", "", "a" * 65):
        with pytest.raises(Phase17MoveError, match="run id"):
            require_run_id(bad)
    assert require_run_id("RUN-2026-A") == "RUN-2026-A"


def test_the_contract_document_is_stable_and_serializable():
    document = move_contract_document()
    assert document["versions"]["transition"] == MOVE_TRANSITION_VERSION
    assert document["objective"]["belief_loss_weight"] == 0.0
    assert document["objective"]["phase9_accepted_belief_loss_weight"] == 0.25
    assert document["kl_controller"]["direction"].startswith("FORWARD")
    assert document["population"]["search"] == "prohibited from collection and training"
    assert document["targets"]["governing_invariant"].startswith("G-M4a")
    assert json.loads(json.dumps(document)) == document
    assert contract_digest() == contract_digest()
    assert len(contract_digest()) == 64


def test_the_seed_domains_are_disjoint_from_phase16():
    from stratego.training.phase16.contract import DOMAIN_ROOTS as PHASE16_ROOTS
    from stratego.training.phase17.move_contract import MOVE_DOMAIN_ROOTS

    assert not set(MOVE_DOMAIN_ROOTS.values()) & set(PHASE16_ROOTS.values())


def test_a_phase17_game_draws_different_actions_than_a_phase16_game():
    """The same slot and draw number must not replay a Phase 16 game."""
    from stratego.training.phase16.collector import action_sampling_uniform as p16
    from stratego.training.phase17.move_snapshot import action_sampling_uniform as p17

    identifier = game_id("RUN-2026-A", 0, 0)
    assert [p16(identifier, ply) for ply in range(20)] != [
        p17(identifier, ply) for ply in range(20)
    ]
