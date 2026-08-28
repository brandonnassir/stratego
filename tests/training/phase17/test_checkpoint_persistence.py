"""Agent 4: the paired checkpoint's atomicity, fail-closed loading, and the
exact active-game persistence common contract section 10 requires.

The expensive part -- a real save/load continuation-equivalence run -- lives in
`test_runner_tandem.py`, which already has a live runner. These tests cover the
file mechanics and the single-game capture/restore in isolation.
"""

from __future__ import annotations

import json

import pytest
import torch

from stratego.engine.snapshot import create_snapshot
from stratego.training.phase17.checkpoint import (
    ACTIVE_GAME_SNAPSHOT_KEYS,
    JOINT_CHECKPOINT_SCHEMA_VERSION,
    Phase17CheckpointError,
    checkpoint_schema,
    json_digest,
    read_joint_checkpoint,
    write_joint_checkpoint,
)
from stratego.training.phase17.checkpoint import REQUIRED_KEYS

RUN_ID = "RUN-TEST-A"


def minimal_payload(**overrides) -> dict:
    payload = {key: None for key in REQUIRED_KEYS}
    payload.update(
        {
            "schema_version": JOINT_CHECKPOINT_SCHEMA_VERSION,
            "run_id": RUN_ID,
            "work_package": "phase17",
            "iteration": 4,
            "move_scheduler_position": {"iteration": 4},
            "setup_scheduler_position": {"iteration": 3},
            "active_games": [{"game_id": "g1"}],
            "active_game_setup_episodes": {"g1": {}},
            "config_digest": "cfg",
            "source_digest": "src",
            "checkpoint_generation": 2,
            "elapsed_active_training_seconds": 12.5,
            "move_raw_state": {"w": torch.zeros(2)},
        }
    )
    payload.update(overrides)
    return payload


# -- atomic write -----------------------------------------------------------


def test_a_written_checkpoint_reproduces_its_own_digest(tmp_path):
    identity = write_joint_checkpoint(minimal_payload(), tmp_path / "joint.pt")
    assert identity.generation == 2 and identity.iteration == 4
    reread = torch.load(tmp_path / "joint.pt", map_location="cpu", weights_only=False)
    assert reread["payload_digest"] == identity.payload_digest


def test_no_partial_file_is_left_under_the_final_name(tmp_path):
    target = tmp_path / "joint.pt"
    write_joint_checkpoint(minimal_payload(), target)
    assert target.exists()
    assert not list(tmp_path.glob("*.partial")), "a temporary file survived the write"


def test_an_accepted_checkpoint_is_never_overwritten(tmp_path):
    target = tmp_path / "joint.pt"
    write_joint_checkpoint(minimal_payload(), target)
    with pytest.raises(Phase17CheckpointError, match="never overwritten"):
        write_joint_checkpoint(minimal_payload(), target)


def test_a_missing_required_key_is_refused_before_anything_is_written(tmp_path):
    payload = minimal_payload()
    del payload["rng_namespaces"]
    with pytest.raises(Phase17CheckpointError, match="missing required key"):
        write_joint_checkpoint(payload, tmp_path / "joint.pt")
    assert not (tmp_path / "joint.pt").exists()


def test_the_digest_does_not_cover_the_field_that_holds_it(tmp_path):
    """The self-referential trap: a payload_digest inside the digested payload."""
    identity = write_joint_checkpoint(minimal_payload(), tmp_path / "joint.pt")
    reread = torch.load(tmp_path / "joint.pt", map_location="cpu", weights_only=False)
    assert "payload_digest" in reread
    # Re-reading and re-digesting must agree, which it only can if the digest
    # excludes itself.
    read_joint_checkpoint(tmp_path / "joint.pt", run_id=RUN_ID)


def test_the_digest_excludes_tensors_but_covers_their_digests(tmp_path):
    a = minimal_payload(move_raw_model_state_digest="aaa")
    b = minimal_payload(move_raw_model_state_digest="bbb")
    assert write_joint_checkpoint(a, tmp_path / "a.pt").payload_digest != (
        write_joint_checkpoint(b, tmp_path / "b.pt").payload_digest
    )


# -- fail-closed loading ----------------------------------------------------


def test_a_foreign_run_is_refused(tmp_path):
    write_joint_checkpoint(minimal_payload(), tmp_path / "joint.pt")
    with pytest.raises(Phase17CheckpointError, match="belongs to run"):
        read_joint_checkpoint(tmp_path / "joint.pt", run_id="RUN-OTHER-A")


def test_a_foreign_config_digest_is_refused(tmp_path):
    write_joint_checkpoint(minimal_payload(), tmp_path / "joint.pt")
    with pytest.raises(Phase17CheckpointError, match="written under config"):
        read_joint_checkpoint(
            tmp_path / "joint.pt", run_id=RUN_ID, config_digest="different"
        )


def test_a_foreign_source_digest_is_refused(tmp_path):
    write_joint_checkpoint(minimal_payload(), tmp_path / "joint.pt")
    with pytest.raises(Phase17CheckpointError, match="written under source"):
        read_joint_checkpoint(
            tmp_path / "joint.pt", run_id=RUN_ID, source_digest="different"
        )


def test_a_tampered_payload_is_refused(tmp_path):
    target = tmp_path / "joint.pt"
    write_joint_checkpoint(minimal_payload(), target)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    payload["elapsed_active_training_seconds"] = 99999.0
    torch.save(payload, target)
    with pytest.raises(Phase17CheckpointError, match="digests to"):
        read_joint_checkpoint(target, run_id=RUN_ID)


def test_halves_that_disagree_on_iteration_are_refused(tmp_path):
    payload = minimal_payload(move_scheduler_position={"iteration": 9})
    write_joint_checkpoint(payload, tmp_path / "joint.pt")
    with pytest.raises(Phase17CheckpointError, match="the move half is at iteration"):
        read_joint_checkpoint(tmp_path / "joint.pt", run_id=RUN_ID)


def test_a_setup_half_ahead_of_the_move_half_is_refused(tmp_path):
    payload = minimal_payload(setup_scheduler_position={"iteration": 5})
    write_joint_checkpoint(payload, tmp_path / "joint.pt")
    with pytest.raises(Phase17CheckpointError, match="ahead of the move half"):
        read_joint_checkpoint(tmp_path / "joint.pt", run_id=RUN_ID)


def test_an_orphaned_setup_episode_is_refused(tmp_path):
    payload = minimal_payload(active_game_setup_episodes={"g1": {}, "ghost": {}})
    write_joint_checkpoint(payload, tmp_path / "joint.pt")
    with pytest.raises(Phase17CheckpointError, match="orphaned"):
        read_joint_checkpoint(tmp_path / "joint.pt", run_id=RUN_ID)


def test_an_active_game_with_no_setup_episode_is_refused(tmp_path):
    payload = minimal_payload(
        active_games=[{"game_id": "g1"}, {"game_id": "g2"}]
    )
    write_joint_checkpoint(payload, tmp_path / "joint.pt")
    with pytest.raises(Phase17CheckpointError, match="missing="):
        read_joint_checkpoint(tmp_path / "joint.pt", run_id=RUN_ID)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(Phase17CheckpointError, match="no paired checkpoint"):
        read_joint_checkpoint(tmp_path / "nothing.pt", run_id=RUN_ID)


# -- the engine snapshot codec ---------------------------------------------


def test_the_captured_snapshot_keys_still_match_the_engine():
    """If the engine grows a state field, this fails rather than dropping it."""
    from stratego.engine.state import create_game
    from stratego.training.warmstart_contract import CORPUS_RULES
    from stratego.training.setup_source import UniformRandomSetupSource

    source = UniformRandomSetupSource()
    assignment = source.assign(
        root_seed=5, slot_seed=5, environment_id=0, generation=0, game_id="g-keys"
    )
    state = create_game(
        assignment.red_setup, assignment.blue_setup, rules=CORPUS_RULES, game_id="g-keys"
    )
    snapshot = create_snapshot(state, include_history=True)
    assert set(snapshot) == set(ACTIVE_GAME_SNAPSHOT_KEYS)


def test_the_schema_document_names_the_resume_divergence_limitation():
    limitation = checkpoint_schema()["known_limitation"]
    assert limitation["field"] == "divergence_rows_lost_to_resume"
    assert "non-gating" in limitation["detail"]


def test_json_digest_is_order_independent():
    assert json_digest({"a": 1, "b": 2}) == json_digest({"b": 2, "a": 1})
    assert json_digest({"a": 1}) != json_digest({"a": 2})
