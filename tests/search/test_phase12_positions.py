"""Phase 12 diagnostic position set: identity, eligibility, selection, replay.

These tests do not play games — a diagnostic game needs the accepted Phase 9
inference owner and belongs to the runner. What is tested here is everything
that decides *which* positions the diagnostic ends up with and whether a
manifest can be trusted to rebuild them: the id grammar, the seed streams,
the eligibility rule, the midpoint selection rule, the manifest digest, and
the bit-for-bit replay check.
"""

import json

import numpy as np
import pytest

from stratego.belief.phase11b.contract import CORPUS_STRATA
from stratego.engine.constants import BLUE, RED
from stratego.engine.legal_moves import legal_actions
from stratego.engine.observation import build_observation
from stratego.engine.state import create_game
from stratego.engine.transition import apply_action
from stratego.evaluation.match_spec import EVALUATION_RULES
from stratego.search.phase12 import positions as diag

from tests.helpers import full_inventory_setup


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_game_id_round_trips_through_its_grammar():
    game_id = diag.diagnostic_game_id("tactical_rule", "p10d", "red", 7)
    assert game_id.startswith("phase12_diag_v1|")
    fields = diag.parse_diagnostic_game_id(game_id)
    assert fields == {
        "master_seed": diag.DIAGNOSTIC_MASTER_SEED,
        "stratum": "tactical_rule",
        "setup_source": "p10d",
        "observer_color": "red",
        "ordinal": 7,
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ("no_such_stratum", "p10d", "red", 0),
        ("tactical_rule", "no_such_source", "red", 0),
        ("tactical_rule", "p10d", "green", 0),
        ("tactical_rule", "p10d", "red", -1),
        ("tactical_rule", "p10d", "red", 10_000),
    ],
)
def test_game_id_refuses_identities_outside_the_grammar(arguments):
    with pytest.raises(diag.Phase12PositionError):
        diag.diagnostic_game_id(*arguments)


def test_the_four_groups_are_the_accepted_strata():
    assert diag.DIAGNOSTIC_STRATA == tuple(CORPUS_STRATA)
    assert len(diag.DIAGNOSTIC_STRATA) == 4
    assert len(diag.DIAGNOSTIC_CELLS) == 16
    # Cell-major balance: every (source, colour) appears once per stratum.
    for stratum in diag.DIAGNOSTIC_STRATA:
        cells = [cell for cell in diag.DIAGNOSTIC_CELLS if cell[0] == stratum]
        assert len(cells) == 4
        assert len({cell[1] for cell in cells}) == 2
        assert len({cell[2] for cell in cells}) == 2


def test_the_target_shape_is_the_instructed_one():
    assert diag.POSITIONS_PER_CELL * len(diag.DIAGNOSTIC_CELLS) == 256
    assert diag.POSITIONS_PER_CELL * 4 == 64  # per behaviour group


def test_seeds_are_deterministic_distinct_and_non_negative():
    game_id = diag.diagnostic_game_id("scout_rush", "neutral", "blue", 3)
    other = diag.diagnostic_game_id("scout_rush", "neutral", "blue", 4)
    values = {
        domain: diag.diagnostic_seed(domain, game_id)
        for domain in (
            diag.DOMAIN_OBSERVER_SETUP,
            diag.DOMAIN_OPPONENT_SETUP,
            diag.DOMAIN_MATCH,
        )
    }
    assert len(set(values.values())) == 3, "the streams must not coincide"
    assert all(0 <= value < 2**63 for value in values.values())
    assert values[diag.DOMAIN_MATCH] == diag.diagnostic_seed(
        diag.DOMAIN_MATCH, game_id
    )
    assert values[diag.DOMAIN_MATCH] != diag.diagnostic_seed(
        diag.DOMAIN_MATCH, other
    )


def test_the_search_seed_depends_only_on_the_position():
    identifier = diag.position_id(
        diag.diagnostic_game_id("strategic_rule", "p10d", "red", 1), 40
    )
    assert diag.search_seed_for(identifier) == diag.search_seed_for(identifier)
    assert diag.search_seed_for(identifier) != diag.search_seed_for(
        diag.position_id(
            diag.diagnostic_game_id("strategic_rule", "p10d", "red", 1), 42
        )
    )


def test_the_library_split_is_neither_the_spent_pool_nor_the_training_pool():
    assert diag.DIAGNOSTIC_LIBRARY_SPLIT == "validation"
    assert diag.DIAGNOSTIC_LIBRARY_SPLIT != "test"
    assert diag.DIAGNOSTIC_LIBRARY_SPLIT != "train"


# ---------------------------------------------------------------------------
# Eligibility and selection
# ---------------------------------------------------------------------------


def _decision(ply: int, unresolved: int) -> dict:
    return {
        "ply": ply,
        "unresolved": unresolved,
        "observation_sha256": "0" * 64,
        "moved_hidden": 1,
        "legal_actions": 10,
    }


def test_eligibility_applies_both_floors():
    decisions = [
        _decision(0, 40),  # too early
        _decision(diag.MIN_PLY - 1, 40),  # still too early
        _decision(diag.MIN_PLY, diag.MIN_UNRESOLVED - 1),  # too little hidden left
        _decision(diag.MIN_PLY, diag.MIN_UNRESOLVED),  # the first eligible one
        _decision(200, 12),
    ]
    eligible = diag.eligible_decisions(decisions)
    assert [row["ply"] for row in eligible] == [diag.MIN_PLY, 200]


def test_selection_takes_quantile_midpoints_not_endpoints():
    values = list(range(100))
    assert diag.spread(values, 2) == [25, 75]
    assert diag.spread(values, 4) == [12, 37, 62, 87]
    # Never the ply floor, never the final decision — that is the point.
    assert 0 not in diag.spread(values, 2)
    assert 99 not in diag.spread(values, 2)


def test_selection_degenerates_gracefully():
    assert diag.spread([], 4) == []
    assert diag.spread([1, 2], 0) == []
    assert diag.spread([1, 2], 5) == [1, 2]
    assert len(diag.spread(list(range(10)), 3)) == 3


def test_select_positions_composes_eligibility_and_spread():
    decisions = [_decision(ply, 30) for ply in range(0, 100, 2)]
    chosen = diag.select_positions(decisions, per_game=2)
    assert len(chosen) == 2
    assert all(row["ply"] >= diag.MIN_PLY for row in chosen)
    assert chosen[0]["ply"] < chosen[1]["ply"]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _short_game(plies: int):
    """A real game prefix and the actions that produce it."""
    red = full_inventory_setup()
    blue = full_inventory_setup()
    state = create_game(red, blue, rules=EVALUATION_RULES, game_id="replay-fixture")
    history = []
    while not state.terminal and len(history) < plies:
        action = legal_actions(state)[0]
        history.append(int(action))
        apply_action(state, action)
    return red, blue, history


def test_replay_rebuilds_the_requested_plies_independently():
    red, blue, history = _short_game(12)
    rebuilt = diag.replay_positions(
        "replay-fixture", red, blue, history, RED, [4, 8]
    )
    assert [entry["ply"] for entry in rebuilt] == [4, 8]
    for entry in rebuilt:
        assert entry["state"].total_moves == entry["ply"]
        assert entry["state"].acting_player == RED
        assert np.array_equal(
            entry["observation"], build_observation(entry["state"], RED)
        )
    # Independent objects: advancing one must not disturb the other.
    first, second = rebuilt
    apply_action(first["state"], legal_actions(first["state"])[0])
    assert second["state"].total_moves == 8


def test_replay_refuses_a_ply_the_history_never_reaches():
    red, blue, history = _short_game(6)
    with pytest.raises(diag.Phase12PositionError):
        diag.replay_positions("replay-fixture", red, blue, history, RED, [40])


def test_replay_refuses_a_ply_the_observer_does_not_act_on():
    red, blue, history = _short_game(8)
    with pytest.raises(diag.Phase12PositionError):
        diag.replay_positions("replay-fixture", red, blue, history, RED, [3])


def test_observation_digest_is_the_corpus_recipe():
    red, blue, history = _short_game(4)
    rebuilt = diag.replay_positions("replay-fixture", red, blue, history, RED, [2])
    import hashlib

    expected = hashlib.sha256(
        np.ascontiguousarray(rebuilt[0]["observation"], dtype=np.float32).tobytes()
    ).hexdigest()
    assert diag.observation_digest(rebuilt[0]["observation"]) == expected


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def _tiny_manifest():
    red, blue, history = _short_game(12)
    game_id = diag.diagnostic_game_id("tactical_rule", "p10d", "red", 0)
    rebuilt = diag.replay_positions(game_id, red, blue, history, RED, [4, 8])
    generated = {
        "games": [
            {
                "game_id": game_id,
                "stratum": "tactical_rule",
                "setup_source": "p10d",
                "observer_color": "red",
                "observer_player": int(RED),
                "ordinal": 0,
                "match_seed": diag.diagnostic_seed(diag.DOMAIN_MATCH, game_id),
                "opponent_policy_id": "test",
                "red_setup": [int(value) for value in red],
                "blue_setup": [int(value) for value in blue],
                "plies": len(history),
                "observer_decisions": 6,
                "eligible_decisions": 2,
                "contributed": 2,
                "action_history": history,
            }
        ],
        "positions": [
            {
                "position_id": diag.position_id(game_id, entry["ply"]),
                "game_id": game_id,
                "ply": entry["ply"],
                "stratum": "tactical_rule",
                "setup_source": "p10d",
                "observer_color": "red",
                "observer_player": int(RED),
                "unresolved": 40,
                "moved_hidden": 1,
                "legal_actions": 10,
                "observation_sha256": diag.observation_digest(entry["observation"]),
            }
            for entry in rebuilt
        ],
        "shortfalls": [],
    }
    return diag.build_manifest(generated, generated_utc="2026-08-20T00:00:00Z")


def test_manifest_carries_its_counts_and_a_content_digest():
    manifest = _tiny_manifest()
    assert manifest["artifact"] == diag.DIAGNOSTIC_VERSION
    assert manifest["counts"]["positions"] == 2
    assert manifest["counts"]["positions_by_behavior_group"] == {"tactical_rule": 2}
    assert manifest["counts"]["positions_by_observer_color"] == {"red": 2}
    assert manifest["phase11_test_bank_used"] is False
    assert manifest["setup_library_split"] == "validation"
    assert manifest["manifest_digest"] == diag.manifest_digest(manifest)


def test_manifest_digest_notices_a_changed_position(tmp_path):
    manifest = _tiny_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert diag.load_manifest(path)["manifest_digest"] == manifest["manifest_digest"]

    tampered = json.loads(path.read_text())
    tampered["positions"][0]["ply"] = 6
    path.write_text(json.dumps(tampered))
    with pytest.raises(diag.Phase12PositionError):
        diag.load_manifest(path)


def test_load_manifest_refuses_a_foreign_artifact(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"artifact": "something_else"}))
    with pytest.raises(diag.Phase12PositionError):
        diag.load_manifest(path)


def test_materialize_rebuilds_playable_states_and_checks_the_digest():
    manifest = _tiny_manifest()
    materialized = diag.materialize_manifest(manifest, verify=True)
    assert len(materialized) == 2
    for record in materialized:
        assert record["state"].total_moves == record["ply"]
        assert record["legal_action_count"] == len(legal_actions(record["state"]))
        assert record["search_seed"] == diag.search_seed_for(record["position_id"])
        assert record["document"]["document_version"] == "phase11_public_state_v1"
        assert record["document_summary"]["progress_bucket"] in (
            "early",
            "middle",
            "late",
        )


def test_materialize_refuses_a_position_that_replays_to_something_else():
    manifest = _tiny_manifest()
    manifest["positions"][0]["observation_sha256"] = "f" * 64
    with pytest.raises(diag.Phase12PositionError):
        diag.materialize_manifest(manifest, verify=True)


def test_the_agent_1_search_core_is_not_imported_from_here_by_accident():
    """positions.py may read the Phase 12 contract, but owns no search."""
    import inspect

    source = inspect.getsource(diag)
    assert "from .engine import" not in source
    assert "from .providers import" not in source
