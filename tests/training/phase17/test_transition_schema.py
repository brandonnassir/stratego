"""Phase 17 Agent 2: the `phase17_move_transition_v1` row and its refusals."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from stratego.training.phase17.move_contract import MOVE_TRANSITION_VERSION
from stratego.training.phase17.transition_schema import (
    MoveTransition,
    Phase17TransitionError,
    assert_unique,
    transition_schema_document,
    validate_transition,
)
from stratego.training.phase9_loss import behavior_probability_matrix


def build_row(**overrides) -> MoveTransition:
    row = MoveTransition(
        run_id="RUN-TEST-A",
        iteration=3,
        window_index=3,
        game_id="phase17_game_v1|run=RUN-TEST-A|slot=0000|draw=000000",
        ply=7,
        color=0,
        perspective_player=0,
        observation=np.zeros((127, 10, 10), dtype=np.float32),
        legal_mask=np.zeros(10000, dtype=bool),
        legal_actions=(4, 19, 300),
        behavior_probabilities=(0.2, 0.3, 0.5),
        sampled_action=19,
        sampled_action_index=1,
        sampled_action_model=19,
        behavior_action_probability=0.3,
        behavior_action_logprob=float(np.log(0.3)),
        action_seed=99,
        behavior_model_state_digest="a" * 64,
        behavior_snapshot_iteration=3,
        stored_value_scalar=0.1,
        stored_wdl=(0.4, 0.3, 0.3),
        wdl_target=(0.5, 0.25, 0.25),
    )
    for name, value in overrides.items():
        setattr(row, name, value)
    return row


def test_a_well_formed_row_validates():
    row = build_row()
    validate_transition(row)
    assert row.schema_version == MOVE_TRANSITION_VERSION
    assert row.key == (row.game_id, 0, 7)


def test_the_identity_row_carries_no_tensors_and_is_serializable():
    import json

    identity = build_row().identity_row()
    assert "observation" not in identity
    assert "legal_mask" not in identity
    assert identity["legal_actions"] == 3
    assert json.loads(json.dumps(identity)) == identity


def test_a_row_with_no_behavior_digest_is_refused():
    with pytest.raises(Phase17TransitionError, match="which policy produced it"):
        validate_transition(build_row(behavior_model_state_digest=""))


def test_a_non_ascending_legal_list_is_refused():
    with pytest.raises(Phase17TransitionError, match="ASCENDING"):
        validate_transition(build_row(legal_actions=(300, 19, 4)))


def test_a_distribution_that_does_not_sum_to_one_is_refused():
    with pytest.raises(Phase17TransitionError, match="sums to"):
        validate_transition(build_row(behavior_probabilities=(0.2, 0.3, 0.1)))


def test_a_negative_probability_is_refused():
    with pytest.raises(Phase17TransitionError, match="negative or non-finite"):
        validate_transition(build_row(behavior_probabilities=(0.7, -0.2, 0.5)))


def test_a_sampled_action_that_is_not_its_index_is_refused():
    with pytest.raises(Phase17TransitionError, match="is not"):
        validate_transition(build_row(sampled_action=4))


def test_an_index_outside_the_legal_set_is_refused():
    with pytest.raises(Phase17TransitionError, match="outside"):
        validate_transition(build_row(sampled_action_index=5))


def test_a_perspective_that_is_not_the_actor_is_refused():
    """Both seats are learners, so the two can never legitimately differ."""
    with pytest.raises(Phase17TransitionError, match="perspective_player must equal"):
        validate_transition(build_row(perspective_player=1))


def test_an_unknown_boundary_status_or_provenance_is_refused():
    with pytest.raises(Phase17TransitionError, match="boundary_status"):
        validate_transition(build_row(boundary_status="halfway"))
    with pytest.raises(Phase17TransitionError, match="target_provenance"):
        validate_transition(build_row(target_provenance="guessed"))


def test_a_wdl_target_off_the_simplex_is_refused():
    with pytest.raises(Phase17TransitionError, match="W/D/L target"):
        validate_transition(build_row(wdl_target=(0.5, 0.5, 0.5)))
    with pytest.raises(Phase17TransitionError, match="simplex"):
        validate_transition(build_row(wdl_target=(1.5, -0.5, 0.0)))


def test_a_non_finite_advantage_is_refused():
    with pytest.raises(Phase17TransitionError, match="non-finite advantage"):
        validate_transition(build_row(advantage_target=float("inf")))


def test_a_negative_age_is_refused():
    with pytest.raises(Phase17TransitionError, match="policy_age_iterations"):
        validate_transition(build_row(policy_age_iterations=-1))
    with pytest.raises(Phase17TransitionError, match="bootstrap_age_windows"):
        validate_transition(build_row(bootstrap_age_windows=-2))


def test_a_row_of_the_wrong_type_is_refused():
    with pytest.raises(Phase17TransitionError, match="expected a MoveTransition"):
        validate_transition({"game_id": "g"})


def test_the_same_transition_emitted_twice_is_caught():
    first = build_row()
    second = copy.copy(first)
    second.window_index = 4
    with pytest.raises(Phase17TransitionError, match="emitted twice"):
        assert_unique([first, second])
    assert assert_unique([first, build_row(ply=8)]) == {"rows": 2, "duplicates": 0}


def test_the_accepted_dense_builder_reads_the_row_unchanged():
    """The aliases exist so the accepted collation is reused, not forked."""
    row = build_row()
    matrix = behavior_probability_matrix([row])
    assert matrix.shape == (1, 10000)
    assert float(matrix.sum()) == pytest.approx(1.0, abs=1e-6)
    assert row.learner_side == row.color
    assert row.decision_index == row.ply
    assert row.behavior_legal_actions == row.legal_actions
    assert row.behavior_legal_probabilities == row.behavior_probabilities


def test_the_advantage_alias_reads_and_writes_the_target():
    row = build_row()
    row.advantage = -0.25
    assert row.advantage_target == pytest.approx(-0.25)
    assert row.advantage == pytest.approx(-0.25)


def test_the_schema_document_names_the_telemetry_fields():
    document = transition_schema_document()
    assert document["unique_under"] == ["game_id", "color", "ply"]
    assert "boundary_target_divergence" in document["telemetry_never_a_gate"]
    assert "bootstrap_age_windows" in document["telemetry_never_a_gate"]
    assert document["current_policy_proof"] == "behavior_model_state_digest"


def test_the_row_carries_no_privileged_label_at_all():
    """No belief target, no enemy identity, no future action.

    Phase 16's harvested row carried `belief_target` / `belief_mask` built from
    `dense_belief_target(state, actor)` -- privileged truth, legitimate there
    because Phase 9's objective had a belief term. Phase 17 disables that term,
    so the label is not built rather than built and weighted to zero: there is
    nothing in the row a leak could come from.
    """
    fields = set(MoveTransition.__dataclass_fields__)
    assert not any("belief" in name for name in fields)
    assert not any("enemy" in name or "truth" in name for name in fields)

    row = build_row()
    tensors = [
        name
        for name in fields
        if isinstance(getattr(row, name), np.ndarray)
    ]
    assert sorted(tensors) == ["legal_mask", "observation"]


def test_every_stored_field_is_knowable_at_the_moment_of_the_decision():
    """Except the four the window and the game's end legitimately fill in."""
    filled_later = {
        # the window close fills these from the segment's tail
        "advantage_target",
        "wdl_target",
        "target_provenance",
        "boundary_status",
        "bootstrap_age_windows",
        "standardized_advantage",
        "ppo_eligible",
        "value_row_weight",
        "policy_age_iterations",
        "iteration",
        "window_index",
        # the game's end fills these, as telemetry only
        "boundary_target_divergence",
        "boundary_wdl_divergence",
    }
    at_decision = set(MoveTransition.__dataclass_fields__) - filled_later
    # everything at decision time is either identity, the stored decision, or a
    # frozen contract version -- and none of it depends on the future.
    assert "sampled_action" in at_decision
    assert "behavior_probabilities" in at_decision
    assert "behavior_model_state_digest" in at_decision
    assert "stored_wdl" in at_decision
    assert not (at_decision & filled_later)
