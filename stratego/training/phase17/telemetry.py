"""Phase 17 Agent 4: the durable append-only tandem telemetry log.

Specification sources: Agent 4 instruction section 8, common contract sections
6, 8 and 13.

Durability, and why the append position is checkpointed
-------------------------------------------------------
Rows are JSONL, one line per tandem iteration, `flush()` + `os.fsync()` before
the write is called done. A resumed run must not append a second copy of the
iteration it was interrupted after, and it must not silently start a *new* log
alongside the old one. Both are prevented the same way: the paired checkpoint
carries the byte offset the log had reached and the digest of the last durable
record, and :meth:`TelemetryWriter.resume` refuses to continue a file whose
tail does not reproduce that digest at that offset. A truncated tail is
truncated back to the last verified record rather than appended past.

`last_record_digest` is the json-document digest of the row itself, which is
what a later reader can recompute. The offset alone would not detect a file
that had been replaced by a different log of the same length.

The pending row, and why the checkpointed iteration keeps its row
------------------------------------------------------------------
A session checkpoints *before* it appends the row for that iteration, because
the row carries the written checkpoint's verified identity and that identity
does not exist until the file has landed. Recording only the pre-append
position therefore used to cost one telemetry row per resume: the checkpoint
said "N rows", the row for iteration N+1 was appended, and the resumed writer
truncated it back as excess -- deleting the record of an iteration whose
weights had been checkpointed and were about to be restored.

So the position also names the row it is about to write. `pending_row_iteration`
is the iteration whose row follows this checkpoint; on resume exactly one
record past the offset is adopted, and only if it is that row: right run,
right record index, right `system.iteration`, and a complete line. Anything
else past the offset is genuinely uncheckpointed work from a later iteration
this resume is discarding, and is truncated as before.

Schema
------
The row schema is the one Agent 1 froze for the checkpoint and episode
documents, extended per Agent 4 instruction section 8 into the three named
blocks the instruction lists: `move`, `setup`, `system`. Required keys are
asserted on every write, so a missing telemetry field fails at the iteration
that dropped it rather than at closeout.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .move_contract import Phase17MoveError

TELEMETRY_SCHEMA_VERSION = "phase17_simple_tandem_telemetry_v1"

#: Top-level blocks every row carries.
REQUIRED_BLOCKS = ("move", "setup", "system")

#: Keys required inside each block. Absence fails closed (Agent 1's encoding
#: rules: "a required field that is absent fails closed").
REQUIRED_MOVE_KEYS = (
    "transitions_harvested",
    "transitions_trained",
    "boundary_rows",
    "terminal_rows",
    "active_games",
    "games_completed",
    "game_lengths",
    "terminal_results",
    "terminal_reasons",
    "plies_advanced",
    "loss_components",
    "entropy",
    "mean_kl",
    "kl_beta",
    "clip_fraction",
    "grad_norm",
    "learning_rate",
    "entropy_coefficient",
    "raw_model_state_digest",
    "ema_model_state_digest",
    "optimizer_steps",
    "participant_ledger",
    "boundary_target_divergence",
    "bootstrap_age_windows",
    "collection_seconds",
    "target_seconds",
    "optimization_seconds",
)
#: The setup block names a fixed `kl_coefficient`, never a `kl_beta`: operator
#: decision D10 section 1 requires telemetry to call the reverse-KL weight what
#: it is, so no reader can mistake a constant for a regulated quantity.
REQUIRED_SETUP_KEYS = (
    "generated",
    "refills",
    "unused",
    "discarded_on_rebind",
    "snapshot_iteration",
    "raw_model_state_digest",
    "ema_model_state_digest",
    "legality_failures",
    "orientation_failures",
    "fallback_attempts",
    "completed_episode_buffer",
    "activity",
    "updated",
    "skip_reason",
    "episodes_consumed",
    "loss_components",
    "advantage_components",
    "empirical_entropy",
    "predicted_entropy",
    "mean_kl",
    "final_epoch_kl",
    "kl_coefficient",
    "kl_direction",
    "grad_norm",
    "learning_rate",
    "alpha",
    "optimizer_steps",
    "concentration",
    "generation_seconds",
    "optimization_seconds",
)
REQUIRED_SYSTEM_KEYS = (
    "run_id",
    "work_package",
    "iteration",
    "elapsed_active_training_seconds",
    "cadence_index",
    "memory_high_water_mib",
    "checkpoint",
    "export",
    "warnings",
    "stop_predicates",
    "external_result_status",
    "checkpoint_seconds",
    "total_seconds",
)


class Phase17TelemetryError(Phase17MoveError):
    """A telemetry row or log could not be written or continued as specified."""


def row_digest(row: dict) -> str:
    """The accepted json-document digest of one telemetry row."""
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_row(row: dict) -> None:
    """Fail closed on a row missing any frozen field."""
    missing_blocks = [block for block in REQUIRED_BLOCKS if block not in row]
    if missing_blocks:
        raise Phase17TelemetryError(
            f"telemetry row is missing block(s) {missing_blocks}; the schema is "
            f"{TELEMETRY_SCHEMA_VERSION}"
        )
    for block, required in (
        ("move", REQUIRED_MOVE_KEYS),
        ("setup", REQUIRED_SETUP_KEYS),
        ("system", REQUIRED_SYSTEM_KEYS),
    ):
        payload = row[block]
        if not isinstance(payload, dict):
            raise Phase17TelemetryError(f"telemetry block {block!r} is not a mapping")
        missing = [key for key in required if key not in payload]
        if missing:
            raise Phase17TelemetryError(
                f"telemetry block {block!r} is missing {missing}"
            )


@dataclass
class TelemetryWriter:
    """Append-only JSONL with a checkpointable, verifiable append position."""

    path: Path
    run_id: str
    records: int = 0
    offset: int = 0
    last_record_digest: "str | None" = None
    _handle: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "TelemetryWriter":
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TelemetryWriter":
        return self.open()

    def __exit__(self, *exception) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    def append(self, row: dict) -> dict:
        """Validate, write, flush and fsync one row. Returns its append receipt."""
        validate_row(row)
        stamped = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "record_index": int(self.records),
            **row,
        }
        digest = row_digest(stamped)
        line = json.dumps(stamped, sort_keys=True, separators=(",", ":")) + "\n"
        handle = self.open()._handle
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        self.records += 1
        self.offset += len(line.encode("utf-8"))
        self.last_record_digest = digest
        return {
            "record_index": stamped["record_index"],
            "record_digest": digest,
            "offset": self.offset,
        }

    # -- persistence -------------------------------------------------------

    def position(self, *, pending_row_iteration: "int | None" = None) -> dict:
        """The append position the paired checkpoint carries.

        `pending_row_iteration` is the iteration whose row the caller is about
        to append after this checkpoint lands. Naming it is what lets the
        resume keep that row instead of truncating it as excess.
        """
        return {
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "path": str(self.path),
            "records": int(self.records),
            "offset": int(self.offset),
            "last_record_digest": self.last_record_digest,
            "pending_row_iteration": (
                None if pending_row_iteration is None else int(pending_row_iteration)
            ),
        }

    def _adopt_pending_row(self, tail: bytes, pending_row_iteration: "int | None") -> int:
        """Bytes to keep past the checkpointed offset: one row, or none.

        The row is adopted only if every one of these holds, because each one
        is a way the tail could belong to something else: the line is complete
        (a crash mid-write leaves no trailing newline), it parses, it belongs
        to this run, it sits at exactly the next record index, and it is the
        iteration the checkpoint said was pending. A tail that fails any of
        them is discarded, which is the pre-correction behavior for everything
        that is not the checkpointed iteration's own row.
        """
        if pending_row_iteration is None or not tail:
            return 0
        newline = tail.find(b"\n")
        if newline < 0:
            return 0
        line = tail[: newline + 1]
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return 0
        if not isinstance(record, dict):
            return 0
        if record.get("run_id") != self.run_id:
            return 0
        if record.get("record_index") != self.records:
            return 0
        system = record.get("system")
        if not isinstance(system, dict) or system.get("iteration") != int(
            pending_row_iteration
        ):
            return 0
        self.records += 1
        self.offset += len(line)
        self.last_record_digest = row_digest(record)
        return len(line)

    @classmethod
    def resume(cls, position: dict, *, run_id: str) -> "TelemetryWriter":
        """Continue a log, refusing a tail that does not match the checkpoint.

        A crash between the fsync and the checkpoint leaves the file *longer*
        than the recorded offset. Most of that excess is work this resume is
        discarding -- rows of iterations after the one being restored -- and it
        is truncated back rather than appended past: an append would leave two
        rows for one iteration and a later reader could not tell which one the
        training actually used.

        The one exception is the checkpointed iteration's own row, which the
        session appends *after* the checkpoint lands so it can carry the
        checkpoint's verified identity. That row is named by the position's
        `pending_row_iteration` and adopted, once, before the truncation.
        """
        path = Path(position["path"])
        writer = cls(
            path=path,
            run_id=run_id,
            records=int(position["records"]),
            offset=int(position["offset"]),
            last_record_digest=position.get("last_record_digest"),
        )
        if not path.exists():
            if writer.records:
                raise Phase17TelemetryError(
                    f"the checkpoint records {writer.records} telemetry rows at "
                    f"{path}, which does not exist; refusing to start a second log"
                )
            return writer

        size = path.stat().st_size
        if size < writer.offset:
            raise Phase17TelemetryError(
                f"{path} is {size} bytes but the checkpoint recorded an append "
                f"position of {writer.offset}; the log has been truncated and "
                "the run cannot prove which rows were durable"
            )
        if writer.records:
            with path.open("rb") as handle:
                head = handle.read(writer.offset)
            lines = head.splitlines()
            if len(lines) != writer.records:
                raise Phase17TelemetryError(
                    f"{path} holds {len(lines)} rows before offset {writer.offset}, "
                    f"not the recorded {writer.records}"
                )
            observed = row_digest(json.loads(lines[-1]))
            if observed != writer.last_record_digest:
                raise Phase17TelemetryError(
                    f"{path}: the record at the checkpointed append position "
                    f"digests to {observed}, not the recorded "
                    f"{writer.last_record_digest}; this is not the same log"
                )
        if size > writer.offset:
            with path.open("rb") as handle:
                handle.seek(writer.offset)
                tail = handle.read()
            writer._adopt_pending_row(tail, position.get("pending_row_iteration"))
        if size > writer.offset:
            with path.open("r+b") as handle:
                handle.truncate(writer.offset)
                handle.flush()
                os.fsync(handle.fileno())
        return writer


def read_rows(path: "str | Path") -> list:
    """Every durable row of a telemetry log, in order."""
    target = Path(path)
    if not target.exists():
        return []
    return [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def telemetry_schema() -> dict:
    """The frozen schema, for the run config and the handoff."""
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "format": "JSONL, one row per tandem iteration, fsynced before return",
        "blocks": list(REQUIRED_BLOCKS),
        "move": list(REQUIRED_MOVE_KEYS),
        "setup": list(REQUIRED_SETUP_KEYS),
        "system": list(REQUIRED_SYSTEM_KEYS),
        "append_position": [
            "records",
            "offset",
            "last_record_digest",
            "pending_row_iteration",
        ],
        "resume_rule": (
            "a resumed log must reproduce last_record_digest at the recorded "
            "offset; past that offset the single row named by "
            "pending_row_iteration is adopted if it is complete and matches "
            "run, record index and iteration, and everything else is truncated "
            "back. A shorter log is refused."
        ),
    }


__all__ = [
    "Phase17TelemetryError",
    "REQUIRED_BLOCKS",
    "REQUIRED_MOVE_KEYS",
    "REQUIRED_SETUP_KEYS",
    "REQUIRED_SYSTEM_KEYS",
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryWriter",
    "read_rows",
    "row_digest",
    "telemetry_schema",
    "validate_row",
]
