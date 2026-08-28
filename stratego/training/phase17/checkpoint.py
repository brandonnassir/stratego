"""Phase 17 Agent 4: the atomic paired move/setup checkpoint.

Specification sources: common contract section 10, Agent 4 instruction section
6, Agent 1's `phase17_joint_checkpoint_v1` schema.

Load completely, or refuse completely
-------------------------------------
Every check runs before anything is restored -- the accepted Phase 9
read/validate/authorize/rebuild order. A checkpoint whose move half and setup
half disagree on run, iteration or config is refused rather than half-loaded,
because a half-loaded pair is a run whose PPO denominator no longer matches the
weights that produced it and nothing downstream would notice.

Exact active-game persistence
-----------------------------
Common contract section 10 requires the active population to survive a resume
*exactly*, and makes Agent 4 stop for operator review if that proves
impossible. It does not prove impossible. Three things have to be carried, and
the engine already knows how to carry all three:

```text
engine state   create_snapshot(state, include_history=True) / restore_snapshot
               -- the engine's own accepted codec, every field, including the
               derived event log and the action history
trajectory     the builder's decisions, per-ply engine snapshots and actions,
               which is what `finish()` seals a valid GameRecord from
target carry   Agent 2's SeatTrace.to_dict()/from_dict()
```

Nothing is replayed and nothing is recomputed. Replaying the action history
would rebuild the engine state correctly but could not rebuild the *builder*,
whose per-decision records hold the behavior probabilities the model produced
at the time -- probabilities a changed model can no longer reproduce. So the
records are stored, not re-derived.

`rules` is deliberately not stored as an object. The snapshot dictionary holds
a frozen `RulesConfig` by reference; a checkpoint that carried it would be a
checkpoint that could silently reintroduce a foreign ruleset. Only the rules
version is stored, and the load rebinds the accepted `CORPUS_RULES` after
checking that version.

What a resumed game *cannot* reproduce
--------------------------------------
Rows already emitted into a previous window are gone: they were trained on and
the window that owned them is closed. `whole_game_divergence` therefore reports
only the post-resume rows of a game that spanned a resume. That is telemetry
which operator decision D2 explicitly made non-gating, and it is recorded in
the checkpoint document as `divergence_rows_lost_to_resume` rather than left to
be discovered later.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from ...engine.snapshot import create_snapshot, restore_snapshot
from ..trajectory import DecisionRecord, SnapshotEntry
from ..warmstart_contract import CORPUS_RULES
from .move_contract import Phase17MoveError
from .transition_targets import SeatTrace

JOINT_CHECKPOINT_SCHEMA_VERSION = "phase17_joint_checkpoint_v1"

#: The snapshot dictionary keys `create_snapshot(include_history=True)` emits.
#: Listed so a future engine field addition fails here rather than silently
#: dropping out of the persisted population.
ACTIVE_GAME_SNAPSHOT_KEYS = (
    "snapshot_version",
    "rules",
    "game_id",
    "board",
    "pieces",
    "acting_player",
    "phase",
    "total_moves",
    "battleless_moves",
    "terminal",
    "terminal_reason",
    "winner",
    "is_draw",
    "recent_moves",
    "active_threat_relations",
    "behavior_memory",
    "events",
    "action_history",
)

REQUIRED_KEYS = (
    "schema_version",
    "run_id",
    "work_package",
    "iteration",
    "start_identity",
    "move_raw_state",
    "move_raw_model_state_digest",
    "move_ema_state",
    "move_ema_model_state_digest",
    "setup_raw_state",
    "setup_raw_model_state_digest",
    "setup_ema_state",
    "setup_ema_model_state_digest",
    "move_optimizer_state",
    "setup_optimizer_state",
    "move_kl_controller_state",
    "setup_kl_controller_state",
    "move_scheduler_position",
    "setup_scheduler_position",
    "move_optimizer_step_count",
    "setup_optimizer_step_count",
    "rng_namespaces",
    "active_games",
    "active_game_setup_episodes",
    "boundary_carry_state",
    "completed_setup_queue",
    "setup_pool_identity",
    "run_digest",
    "config_digest",
    "source_digest",
    "elapsed_active_training_seconds",
    "written_utc",
    # Agent 4 instruction section 6 additions
    "checkpoint_generation",
    "parent_checkpoint_identity",
    "next_export_boundary_seconds",
    "telemetry_position",
    "supervisor_state",
    "collector_counters",
    "move_trainer_state",
    "setup_ema_updates",
    "setup_config_digest",
    "window_partial_state",
)


class Phase17CheckpointError(Phase17MoveError):
    """A paired checkpoint could not be written, verified or restored."""


def json_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def file_sha256(path: "str | Path") -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# One active game
# ---------------------------------------------------------------------------


def capture_active_game(runner, *, slot: int, draw: int) -> dict:
    """Everything needed to resume one in-flight game exactly."""
    snapshot = create_snapshot(runner.state, include_history=True)
    missing = [key for key in ACTIVE_GAME_SNAPSHOT_KEYS if key not in snapshot]
    if missing:
        raise Phase17CheckpointError(
            f"{runner.game_id}: the engine snapshot is missing {missing}; the "
            "active population cannot be persisted exactly and common contract "
            "section 10 requires a stop for operator review"
        )
    rules = snapshot["rules"]
    stored = {key: snapshot[key] for key in ACTIVE_GAME_SNAPSHOT_KEYS if key != "rules"}
    stored["rules_version"] = rules.rules_version

    builder = runner.builder
    return {
        "slot": int(slot),
        "draw": int(draw),
        "game_id": runner.game_id,
        "engine_snapshot": stored,
        "red_setup": list(builder.red_setup),
        "blue_setup": list(builder.blue_setup),
        "setup_root_seed": int(builder.root_seed),
        "environment_id": int(builder.environment_id),
        "generation": int(builder.generation),
        "snapshot_interval": int(builder.snapshot_interval),
        "collection_policy_version": builder.collection_policy_version,
        "collection_checkpoint_id": builder.collection_checkpoint_id,
        "setup_family": builder.setup_family,
        "builder_actions": [int(action) for action in builder._actions],
        "builder_snapshots": [
            {"ply": int(entry.ply), "payload": entry.payload}
            for entry in builder._snapshots
        ],
        "builder_decisions": [
            {
                "game_id": decision.game_id,
                "ply": int(decision.ply),
                "acting_player": int(decision.acting_player),
                "selected_action_id": int(decision.selected_action_id),
                "legal_action_ids": [int(a) for a in decision.legal_action_ids],
                "old_probabilities": [float(p) for p in decision.old_probabilities],
                "win_draw_loss_prediction": [
                    float(v) for v in decision.win_draw_loss_prediction
                ],
                "collection_policy_version": decision.collection_policy_version,
                "snapshot_reference": int(decision.snapshot_reference),
            }
            for decision in builder._decisions
        ],
        "traces": [trace.to_dict() for trace in runner.traces.values()],
        "counters": {
            "learner_decision_count": int(runner.learner_decision_count),
            "neural_decision_count": int(runner.neural_decision_count),
            "learner_neural_decision_count": int(runner.learner_neural_decision_count),
            "rule_decision_count": int(runner.rule_decision_count),
        },
        "emitted_rows_before_this_checkpoint": len(runner.rows),
        "pending_request_dropped": runner.pending is None,
    }


def restore_active_game(runner, payload: dict) -> None:
    """Put one captured game back into a freshly seated runner, exactly."""
    if runner.game_id != payload["game_id"]:
        raise Phase17CheckpointError(
            f"slot {payload['slot']} was re-seated as {runner.game_id!r} but the "
            f"checkpoint holds {payload['game_id']!r}; the draw sequence diverged"
        )
    stored = dict(payload["engine_snapshot"])
    version = stored.pop("rules_version")
    if version != CORPUS_RULES.rules_version:
        raise Phase17CheckpointError(
            f"{runner.game_id}: checkpointed under rules {version!r}, this build "
            f"is {CORPUS_RULES.rules_version!r}"
        )
    runner.state = restore_snapshot({**stored, "rules": CORPUS_RULES})

    builder = runner.builder
    builder._actions = [int(action) for action in payload["builder_actions"]]
    builder._snapshots = [
        SnapshotEntry(ply=int(entry["ply"]), payload=entry["payload"])
        for entry in payload["builder_snapshots"]
    ]
    builder._decisions = [
        DecisionRecord(
            game_id=decision["game_id"],
            ply=int(decision["ply"]),
            acting_player=int(decision["acting_player"]),
            selected_action_id=int(decision["selected_action_id"]),
            legal_action_ids=tuple(int(a) for a in decision["legal_action_ids"]),
            old_probabilities=tuple(float(p) for p in decision["old_probabilities"]),
            win_draw_loss_prediction=tuple(
                float(v) for v in decision["win_draw_loss_prediction"]
            ),
            collection_policy_version=decision["collection_policy_version"],
            snapshot_reference=int(decision["snapshot_reference"]),
        )
        for decision in payload["builder_decisions"]
    ]

    runner.traces = {}
    for entry in payload["traces"]:
        trace = SeatTrace.from_dict(entry)
        runner.traces[int(trace.color)] = trace

    counters = payload["counters"]
    runner.learner_decision_count = int(counters["learner_decision_count"])
    runner.neural_decision_count = int(counters["neural_decision_count"])
    runner.learner_neural_decision_count = int(counters["learner_neural_decision_count"])
    runner.rule_decision_count = int(counters["rule_decision_count"])
    # Rows already emitted belong to closed windows. A resumed game starts a
    # fresh row buffer; the trace's emission cursor is what stops a transition
    # from being emitted twice, and it was restored above.
    runner.rows = []
    runner.row_index_by_key = {}
    runner.pending = None
    runner.record = None


# ---------------------------------------------------------------------------
# The paired checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointIdentity:
    """What a written checkpoint is, for its parent link and its telemetry row."""

    path: str
    generation: int
    iteration: int
    file_sha256: str
    payload_digest: str

    def document(self) -> dict:
        return {
            "path": self.path,
            "generation": int(self.generation),
            "iteration": int(self.iteration),
            "file_sha256": self.file_sha256,
            "payload_digest": self.payload_digest,
        }


def _digestible(payload: dict) -> dict:
    """The identity fields of a checkpoint, without its tensors.

    Tensors are excluded on purpose: their digests are already in the payload
    as `*_model_state_digest`, and hashing several hundred megabytes of weights
    a second time would make every checkpoint cost more than the iteration that
    produced it.
    """
    skip = {
        # The digest cannot cover the field that holds it.
        "payload_digest",
        "move_raw_state",
        "move_ema_state",
        "setup_raw_state",
        "setup_ema_state",
        "move_optimizer_state",
        "setup_optimizer_state",
    }
    return {key: value for key, value in payload.items() if key not in skip}


def write_joint_checkpoint(payload: dict, path: "str | Path") -> CheckpointIdentity:
    """Write a paired checkpoint atomically, then verify what landed.

    write-to-temporary in the same directory, fsync the file, fsync the
    directory, then `os.replace`. A reader never sees a partial file under the
    final name, and an accepted checkpoint is never overwritten.
    """
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise Phase17CheckpointError(
            f"the paired checkpoint is missing required key(s) {missing}"
        )
    if payload["schema_version"] != JOINT_CHECKPOINT_SCHEMA_VERSION:
        raise Phase17CheckpointError(
            f"checkpoint schema {payload['schema_version']!r} is not "
            f"{JOINT_CHECKPOINT_SCHEMA_VERSION!r}"
        )
    target = Path(path)
    if target.exists():
        raise Phase17CheckpointError(
            f"{target} already exists; an accepted checkpoint is never overwritten"
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    digest = json_digest(_digestible(payload))
    stamped = {**payload, "payload_digest": digest}

    handle, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".partial"
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as sink:
            torch.save(stamped, sink)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary_path, target)
        directory = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    reread = torch.load(target, map_location="cpu", weights_only=False)
    if json_digest(_digestible(reread)) != digest:
        raise Phase17CheckpointError(
            f"{target} does not reproduce its own payload digest after writing"
        )
    return CheckpointIdentity(
        path=str(target),
        generation=int(payload["checkpoint_generation"]),
        iteration=int(payload["iteration"]),
        file_sha256=file_sha256(target),
        payload_digest=digest,
    )


def read_joint_checkpoint(
    path: "str | Path",
    *,
    run_id: str,
    config_digest: "str | None" = None,
    source_digest: "str | None" = None,
) -> dict:
    """Read, validate and authorize a paired checkpoint. Nothing partial."""
    target = Path(path)
    if not target.is_file():
        raise Phase17CheckpointError(f"no paired checkpoint at {target}")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise Phase17CheckpointError(f"{target} is not a checkpoint mapping")

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise Phase17CheckpointError(f"{target} is missing required key(s) {missing}")
    if payload["schema_version"] != JOINT_CHECKPOINT_SCHEMA_VERSION:
        raise Phase17CheckpointError(
            f"{target} is schema {payload['schema_version']!r}, not "
            f"{JOINT_CHECKPOINT_SCHEMA_VERSION!r}"
        )
    if payload["run_id"] != run_id:
        raise Phase17CheckpointError(
            f"{target} belongs to run {payload['run_id']!r}, not {run_id!r}"
        )
    recorded = payload.get("payload_digest")
    observed = json_digest(_digestible(payload))
    if recorded != observed:
        raise Phase17CheckpointError(
            f"{target} digests to {observed}, not the recorded {recorded}"
        )
    if config_digest is not None and payload["config_digest"] != config_digest:
        raise Phase17CheckpointError(
            f"{target} was written under config {payload['config_digest']}, this "
            f"process is {config_digest}"
        )
    if source_digest is not None and payload["source_digest"] != source_digest:
        raise Phase17CheckpointError(
            f"{target} was written under source {payload['source_digest']}, this "
            f"process is {source_digest}"
        )

    # The two halves must agree, or the PPO denominator and the weights that
    # produced it come from different runs.
    move_iteration = int(payload["move_scheduler_position"]["iteration"])
    setup_iteration = int(payload["setup_scheduler_position"]["iteration"])
    if move_iteration != int(payload["iteration"]):
        raise Phase17CheckpointError(
            f"{target}: the move half is at iteration {move_iteration} but the "
            f"checkpoint claims {payload['iteration']}"
        )
    if setup_iteration > move_iteration:
        raise Phase17CheckpointError(
            f"{target}: the setup half is at iteration {setup_iteration}, ahead "
            f"of the move half's {move_iteration}"
        )
    active = payload["active_games"]
    episodes = payload["active_game_setup_episodes"]
    seated = {entry["game_id"] for entry in active}
    attached = set(episodes)
    if seated != attached:
        raise Phase17CheckpointError(
            f"{target}: {len(seated)} active games but setup episodes for "
            f"{len(attached)}; missing={sorted(seated - attached)} "
            f"orphaned={sorted(attached - seated)}"
        )
    return payload


def checkpoint_schema() -> dict:
    return {
        "schema_version": JOINT_CHECKPOINT_SCHEMA_VERSION,
        "required_keys": list(REQUIRED_KEYS),
        "atomic_write": "mkstemp in the target directory, fsync file, os.replace, fsync directory",
        "verification": "re-read and re-digest after the rename; never overwrite an accepted file",
        "compatibility": "FAIL CLOSED: nothing is restored until every check passes",
        "active_population": (
            "exact. create_snapshot(include_history=True) for the engine, the "
            "builder's decisions/snapshots/actions for the trajectory, and "
            "SeatTrace documents for the target carry"
        ),
        "known_limitation": {
            "field": "divergence_rows_lost_to_resume",
            "detail": (
                "whole_game_divergence covers only the post-resume rows of a "
                "game that spanned a resume, because the pre-resume rows were "
                "emitted into a closed window. Telemetry only; operator "
                "decision D2 made boundary-target divergence non-gating."
            ),
        },
    }


__all__ = [
    "ACTIVE_GAME_SNAPSHOT_KEYS",
    "CheckpointIdentity",
    "JOINT_CHECKPOINT_SCHEMA_VERSION",
    "Phase17CheckpointError",
    "REQUIRED_KEYS",
    "capture_active_game",
    "checkpoint_schema",
    "file_sha256",
    "json_digest",
    "read_joint_checkpoint",
    "restore_active_game",
    "write_joint_checkpoint",
]
