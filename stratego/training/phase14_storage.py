"""Phase 14: storage layout, the reserve monitor and the retention policy.

Specification source: `02_AGENT_2_FINAL_TRAINING_INTEGRATION.md` section 13,
over the frozen `storage_retention` block.

The layout
----------
Hot resume checkpoints live on fast internal disk under
`checkpoints/phase14/hot`; everything durable — rollout shards, archive
snapshots, logs and candidate evaluations — lives on the external training
volume under `/Volumes/Brandon_Washington/stratego_phase14`. The split is the
frozen one and matters operationally: hot checkpoints are written every 15
minutes and must not compete with shard writes for the external bus.

Full retention is the plan
--------------------------
Agent 1 measured the projection and froze *full raw retention*: ~726 GiB
worst-case against 994 GiB available. Rolling deletion is a **contingency**,
pre-authorized only below 120 GiB free, and even then it may only remove
Phase 14 raw shards that are explicitly marked consumed, disposable and
safe-to-delete. :func:`plan_rolling_deletion` will not return a path that fails
any of those three tests, and :func:`assert_not_project_evidence` refuses any
path outside the Phase 14 rollout tree — the frozen no-deletion rule for
earlier accepted evidence is enforced by refusing to name such a file, not by
remembering not to.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .phase14_contract import (
    DURABLE_ARCHIVE_SUBDIRECTORY,
    EVALUATION_SUBDIRECTORY,
    EXTERNAL_RUN_DIRECTORY,
    EXTERNAL_VOLUME,
    FULL_RAW_RETENTION,
    HOT_CHECKPOINT_DIRECTORY,
    LOG_SUBDIRECTORY,
    NO_DELETION_RULE,
    PHASE14_NAMESPACE,
    ROLLING_DELETION_RULE,
    ROLLING_DELETION_TRIGGER_GIB,
    ROLLOUT_SUBDIRECTORY,
    STORAGE_RESERVE_GIB,
    repository_root,
)

GIB = 1024**3

#: One in this many deleted shard ranges is kept as a representative sample,
#: per the frozen contingency policy.
DELETION_SAMPLE_STRIDE = 16

SHARD_SUFFIX = ".stgshard"
DISPOSABLE_MARK_SUFFIX = ".disposable.json"


class Phase14StorageError(RuntimeError):
    """Raised when a Phase 14 storage request is unsafe or not well formed."""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase14Storage:
    """Every path the run writes, resolved once.

    `external_root` is a parameter rather than a constant so tests can exercise
    the real code against a temporary directory; production passes the frozen
    volume path and :meth:`verify` says whether it is the frozen one.
    """

    external_root: Path
    hot_root: Path

    @staticmethod
    def production() -> "Phase14Storage":
        return Phase14Storage(
            external_root=Path(EXTERNAL_RUN_DIRECTORY),
            hot_root=repository_root() / HOT_CHECKPOINT_DIRECTORY,
        )

    @staticmethod
    def under(root, *, hot_root=None) -> "Phase14Storage":
        """A complete layout under one directory. The test/rehearsal form."""
        root = Path(root)
        return Phase14Storage(
            external_root=root,
            hot_root=Path(hot_root) if hot_root is not None else root / "hot",
        )

    # -- paths -------------------------------------------------------------

    @property
    def rollout_root(self) -> Path:
        return self.external_root / ROLLOUT_SUBDIRECTORY

    @property
    def archive_root(self) -> Path:
        return self.external_root / DURABLE_ARCHIVE_SUBDIRECTORY

    @property
    def log_root(self) -> Path:
        return self.external_root / LOG_SUBDIRECTORY

    @property
    def evaluation_root(self) -> Path:
        return self.external_root / EVALUATION_SUBDIRECTORY

    @property
    def run_state_path(self) -> Path:
        return self.external_root / "phase14_run_state.json"

    def iteration_directory(self, iteration: int) -> Path:
        """Where the accepted store actually put one iteration's bytes.

        Delegated rather than reconstructed: the layout belongs to the accepted
        rollout store, and a second spelling of it here is exactly how a
        retention pass ends up looking at the wrong directory.
        """
        from .phase9_storage import namespace_rollout_directory

        return namespace_rollout_directory(self.rollout_root, PHASE14_NAMESPACE, iteration)

    def prepare(self) -> dict:
        """Create every directory and prove a byte round-trips in each."""
        created = []
        problems = []
        for path in (
            self.external_root,
            self.rollout_root,
            self.archive_root,
            self.log_root,
            self.evaluation_root,
            self.hot_root,
        ):
            existed = path.exists()
            try:
                path.mkdir(parents=True, exist_ok=True)
                probe = path / ".phase14_write_probe"
                probe.write_bytes(b"phase14")
                if probe.read_bytes() != b"phase14":
                    problems.append(f"write probe under {path} did not read back")
                probe.unlink()
            except OSError as error:
                problems.append(f"{path}: {type(error).__name__}: {error}")
            if not existed:
                created.append(str(path))
        return {"created": created, "problems": problems, "ready": not problems}

    def to_dict(self) -> dict:
        return {
            "external_root": str(self.external_root),
            "rollout_root": str(self.rollout_root),
            "archive_root": str(self.archive_root),
            "log_root": str(self.log_root),
            "evaluation_root": str(self.evaluation_root),
            "hot_root": str(self.hot_root),
            "is_production_layout": self.is_production_layout(),
            "full_raw_retention": FULL_RAW_RETENTION,
            "reserve_gib": STORAGE_RESERVE_GIB,
        }

    def is_production_layout(self) -> bool:
        return str(self.external_root) == EXTERNAL_RUN_DIRECTORY

    # -- capacity ----------------------------------------------------------

    def usage(self) -> dict:
        return volume_usage(self.external_root)

    def reserve_status(self) -> dict:
        """Whether the frozen reserve is intact, and what that authorizes."""
        usage = self.usage()
        free_gib = usage["free_gib"]
        breached = free_gib < ROLLING_DELETION_TRIGGER_GIB
        return {
            **usage,
            "reserve_gib": STORAGE_RESERVE_GIB,
            "reserve_breached": breached,
            "rolling_deletion_authorized": breached,
            "policy": ROLLING_DELETION_RULE,
            "no_deletion_rule": NO_DELETION_RULE,
        }

    def storage_state(self) -> dict:
        """The storage half of a hot checkpoint payload."""
        usage = self.usage()
        return {
            "paths": self.to_dict(),
            "free_bytes": usage["free_bytes"],
            "free_gib": usage["free_gib"],
            "used_bytes": usage["used_bytes"],
            "reserve_gib": STORAGE_RESERVE_GIB,
            "retention": "full raw retention; rolling deletion is contingency only",
        }


def volume_usage(path) -> dict:
    """Live capacity facts about the volume holding `path`.

    Measured now, from this machine, rather than copied from a storage report:
    a plan written yesterday is not evidence about today's free space.
    """
    target = Path(path).expanduser()
    probe = target
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    usage = shutil.disk_usage(probe)
    statvfs = os.statvfs(probe)
    mount = probe.resolve()
    while not os.path.ismount(mount) and mount.parent != mount:
        mount = mount.parent
    return {
        "requested_path": str(target),
        "probed_path": str(probe),
        "mount_point": str(mount),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gib": round(usage.free / GIB, 3),
        "total_gib": round(usage.total / GIB, 3),
        "read_only": bool(statvfs.f_flag & os.ST_RDONLY),
        "external_volume_present": Path(EXTERNAL_VOLUME).exists(),
    }


# ---------------------------------------------------------------------------
# The contingency retention policy
# ---------------------------------------------------------------------------


def mark_shards_disposable(iteration_directory, *, iteration: int, reason: str) -> dict:
    """Record that one iteration's raw shards are consumed and disposable.

    Written only after the iteration is COMMITTED — its examples have been
    consumed by every epoch that will ever read them. The mark is a file beside
    the shards rather than a fact in memory, because the deletion may happen
    days later in a different process.
    """
    directory = Path(iteration_directory)
    if not directory.exists():
        raise Phase14StorageError(f"no iteration directory at {directory}")
    mark = directory / f"iteration_{iteration:04d}{DISPOSABLE_MARK_SUFFIX}"
    record = {
        "artifact": "phase14_disposable_mark_v1",
        "iteration": int(iteration),
        "consumed": True,
        "disposable": True,
        "safe_to_delete": True,
        "reason": str(reason),
        "scope": "Phase 14 raw rollout shards only",
    }
    mark.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def read_disposable_mark(iteration_directory, iteration: int) -> "dict | None":
    mark = Path(iteration_directory) / f"iteration_{iteration:04d}{DISPOSABLE_MARK_SUFFIX}"
    if not mark.exists():
        return None
    return json.loads(mark.read_text())


def assert_not_project_evidence(path, storage: Phase14Storage) -> Path:
    """Refuse any path that is not a Phase 14 raw shard under this run's tree.

    The frozen rule is that earlier accepted project evidence is never deleted
    under any condition. The safe way to honour that is to make the deletion
    helper structurally incapable of naming such a file: anything outside
    `<external_root>/rollouts/phase14/...`, and anything that is not a
    `.stgshard`, is refused here before any caller can act on it.
    """
    target = Path(path).resolve()
    root = (storage.rollout_root / PHASE14_NAMESPACE).resolve()
    if not str(target).startswith(str(root) + os.sep):
        raise Phase14StorageError(
            f"{target} is outside the Phase 14 rollout tree {root}; accepted project "
            "evidence is never deleted, including to create storage space"
        )
    if target.suffix != SHARD_SUFFIX:
        raise Phase14StorageError(
            f"{target} is not a Phase 14 raw shard; only consumed shards are disposable"
        )
    return target


def plan_rolling_deletion(storage: Phase14Storage, *, keep_iterations_after: int = 0) -> dict:
    """What the contingency policy would delete, and what it would keep.

    A *plan*, not an action: it is returned for a human or the runner to act on
    after re-checking the reserve, and it names the retained representative
    sample explicitly so the record of what was removed survives the removal.
    """
    status = storage.reserve_status()
    plan = {
        "authorized": bool(status["rolling_deletion_authorized"]),
        "free_gib": status["free_gib"],
        "trigger_gib": ROLLING_DELETION_TRIGGER_GIB,
        "policy": ROLLING_DELETION_RULE,
        "delete": [],
        "retain_sample": [],
        "skipped_unmarked": [],
        "bytes_reclaimable": 0,
    }
    if not plan["authorized"]:
        return plan

    namespace_root = storage.rollout_root / PHASE14_NAMESPACE
    if not namespace_root.exists():
        return plan
    directories = sorted(namespace_root.glob("iteration_*"))
    for index, directory in enumerate(directories):
        try:
            iteration = int(directory.name.split("_", 1)[1])
        except (IndexError, ValueError):  # pragma: no cover - foreign directory
            continue
        if iteration <= int(keep_iterations_after):
            continue
        mark = read_disposable_mark(directory, iteration)
        if not (mark and mark.get("consumed") and mark.get("disposable") and mark.get("safe_to_delete")):
            plan["skipped_unmarked"].append(str(directory))
            continue
        shards = sorted((directory / "shards").glob(f"*{SHARD_SUFFIX}"))
        if index % DELETION_SAMPLE_STRIDE == 0:
            plan["retain_sample"].append(str(directory))
            continue
        for shard in shards:
            assert_not_project_evidence(shard, storage)
            plan["delete"].append(str(shard))
            plan["bytes_reclaimable"] += shard.stat().st_size
    return plan


def execute_rolling_deletion(storage: Phase14Storage, plan: dict) -> dict:
    """Delete exactly what a plan named, re-checking every path first.

    Re-verifying inside the executor means a plan that was tampered with, or
    that has aged past the state it described, cannot delete anything the
    policy would refuse now.
    """
    if not plan.get("authorized"):
        raise Phase14StorageError(
            "rolling deletion is not authorized; the frozen policy permits it only "
            f"below {ROLLING_DELETION_TRIGGER_GIB} GiB free"
        )
    removed: list = []
    freed = 0
    for path in plan.get("delete", []):
        target = assert_not_project_evidence(path, storage)
        if not target.exists():
            continue
        freed += target.stat().st_size
        target.unlink()
        removed.append(str(target))
    return {
        "removed": removed,
        "bytes_freed": freed,
        "retained_sample": list(plan.get("retain_sample", [])),
        "policy": ROLLING_DELETION_RULE,
    }


def storage_semantics() -> dict:
    return {
        "external_volume": EXTERNAL_VOLUME,
        "external_run_directory": EXTERNAL_RUN_DIRECTORY,
        "hot_directory": HOT_CHECKPOINT_DIRECTORY,
        "full_raw_retention": FULL_RAW_RETENTION,
        "reserve_gib": STORAGE_RESERVE_GIB,
        "contingency": ROLLING_DELETION_RULE,
        "sample_stride": DELETION_SAMPLE_STRIDE,
        "no_deletion_rule": NO_DELETION_RULE,
        "deletion_guard": (
            "only .stgshard files under <external>/rollouts/phase14 can be named; "
            "everything else raises"
        ),
    }
