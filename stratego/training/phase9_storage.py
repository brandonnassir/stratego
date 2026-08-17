"""Phase 9 Agent 2: where Phase 9 rollout bytes live — and why that is not identity.

Specification sources:

- `02_AGENT_2_POPULATION_AND_OPPONENT_SCHEDULER.md` ("Handoff to Agent 3")
- `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` ("Artifact namespaces",
  checkpoint contents: "Absolute paths are diagnostic only and must not
  define identity")
- `phase9_contract.rollout_store_schema()["relocation"]`, which freezes the
  redirect as "explicit operator configuration ... the manifest records the
  actual location, and identity is version + digests, never a path — the
  accepted Phase 8 relocation precedent"

Why this is a separate module
-----------------------------
`phase9_schedule` is the logical schedule and imports nothing from here:
it has no path, no environment variable and no filesystem call anywhere in
its derivations. That separation is the structural proof of the rule this
module exists to serve — **a rollout written at one path and copied
byte-for-byte to another is the same rollout**. Version, logical game ids,
payload/metadata digests and commit identities decide identity; the
directory a byte landed in never does.

The resolution order is the accepted Phase 8 precedent
(`synthetic_corpus.default_corpus_root`), reused verbatim rather than
reinvented:

```text
STRATEGO_PHASE9_ROLLOUT_ROOT       explicit per-process override
data/phase9_rollout_root.txt       the recorded durable redirect, if any
data/phase9/rollouts               the contract's repository default
```

A pointer file survives what an environment variable does not: a later agent
that forgets to export anything still finds the production bytes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: The `phase9_rollout_store_v1` root relative to the repository, frozen by
#: the common contract's artifact namespaces.
DEFAULT_PHASE9_ROLLOUT_ROOT = "data/phase9/rollouts"

#: Per-process override of the rollout root.
PHASE9_ROLLOUT_ROOT_ENV = "STRATEGO_PHASE9_ROLLOUT_ROOT"

#: Durable redirect: a one-line file holding the rollout root. Deliberately
#: outside `data/phase9/` so it survives when the rollout bytes do not live
#: there, exactly as `data/warmstart_corpus_root.txt` does for Phase 8.
PHASE9_ROLLOUT_ROOT_POINTER = "data/phase9_rollout_root.txt"

#: The one sentence every consumer of this module has to carry forward.
STORAGE_IDENTITY_RULE = (
    "the resolved rollout root is an operational diagnostic, never an "
    "identity: Phase 9 rollout identity is rollout version + logical game "
    "ids + payload/metadata digests + commit identities, so the same bytes "
    "copied to another volume are the same rollout"
)


class Phase9StorageError(RuntimeError):
    """Raised when a Phase 9 storage location is unusable as configured."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def default_rollout_root() -> Path:
    """Where this installation keeps `phase9_rollout_store_v1` bytes.

    Every consumer — the collector, the trainer, the auditors — should ask
    this rather than assume a path, so a relocation is one recorded fact
    instead of a search-and-replace. Never called from `phase9_schedule`.
    """
    configured = os.environ.get(PHASE9_ROLLOUT_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    pointer = repository_root() / PHASE9_ROLLOUT_ROOT_POINTER
    if pointer.exists():
        recorded = pointer.read_text().strip()
        if recorded:
            return Path(recorded).expanduser()
    return repository_root() / DEFAULT_PHASE9_ROLLOUT_ROOT


def describe_rollout_root() -> dict:
    """Which redirect (if any) chose the rollout root, for the manifest."""
    configured = os.environ.get(PHASE9_ROLLOUT_ROOT_ENV, "").strip()
    pointer = repository_root() / PHASE9_ROLLOUT_ROOT_POINTER
    recorded = pointer.read_text().strip() if pointer.exists() else ""
    if configured:
        source = "environment"
    elif recorded:
        source = "pointer_file"
    else:
        source = "repository_default"
    return {
        "root": str(default_rollout_root()),
        "source": source,
        "environment_variable": PHASE9_ROLLOUT_ROOT_ENV,
        "environment_value": configured,
        "pointer_file": str(pointer),
        "pointer_value": recorded,
        "repository_default": str(repository_root() / DEFAULT_PHASE9_ROLLOUT_ROOT),
        "identity_rule": STORAGE_IDENTITY_RULE,
    }


def namespace_rollout_directory(root, namespace: str, iteration: int) -> Path:
    """The conventional directory of one iteration's rollout bytes.

    A layout convention only. Nothing in the schedule, the digests or the
    commit identities reads it, so Agent 3 may lay bytes out differently
    without changing a single identity.
    """
    if iteration < 1:
        raise Phase9StorageError(f"iteration must be >= 1, got {iteration}")
    return Path(root) / str(namespace) / f"iteration_{iteration:03d}"


# ---------------------------------------------------------------------------
# Capacity and safe-write diagnostics
# ---------------------------------------------------------------------------


def _nearest_existing(path: Path) -> Path:
    """The closest existing ancestor of `path`, for a pre-creation statvfs."""
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate


def volume_diagnostics(path) -> dict:
    """Live capacity/filesystem facts about the volume holding `path`.

    Measured now, from this machine, rather than copied from an older
    storage report. Reported so an operator can judge a write, never folded
    into any Phase 9 identity.
    """
    target = Path(path).expanduser()
    probe = _nearest_existing(target)
    usage = shutil.disk_usage(probe)
    statvfs = os.statvfs(probe)
    mount = probe.resolve()
    while not os.path.ismount(mount) and mount.parent != mount:
        mount = mount.parent
    return {
        "requested_path": str(target),
        "existing_ancestor_probed": str(probe),
        "mount_point": str(mount),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gib": round(usage.free / 1024**3, 3),
        "block_size": int(statvfs.f_frsize),
        "read_only": bool(statvfs.f_flag & os.ST_RDONLY),
        "path_exists": target.exists(),
    }


def check_writable(path) -> dict:
    """Create `path` if needed and prove a byte can be written and removed.

    A capacity number does not prove a write: a read-only mount, a stale
    automount or a permissions problem all report free space happily. This
    performs the actual round trip and cleans up after itself.
    """
    target = Path(path).expanduser()
    created_root = not target.exists()
    problems: list[str] = []
    probe_written = False
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".phase9_write_probe"
        probe.write_bytes(b"phase9")
        probe_written = probe.read_bytes() == b"phase9"
        if not probe_written:
            problems.append(f"write probe under {target} did not read back")
        probe.unlink()
    except OSError as error:
        problems.append(f"{type(error).__name__}: {error}")
    return {
        "path": str(target),
        "writable": probe_written and not problems,
        "directory_created": created_root,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Volume-requirement estimate for the Phase 9 rollout corpus
# ---------------------------------------------------------------------------

#: Measured Phase 8 corpus cost per committed game, over the whole accepted
#: `synthetic_warmstart_corpus_v1`: 352,975,450 bytes / 28,000 games. Phase 9
#: reuses `trajectory_v1` with the same zlib `stgshard_v1` container and the
#: same commit-journal shape, so this is the honest per-game base rate.
PHASE8_MEASURED_BYTES_PER_GAME = 352_975_450 / 28_000

#: Phase 9 rollout games run longer than the Phase 8 rule-vs-rule corpus and
#: carry a per-game `phase9_rollout_store_v1` sidecar, so the base rate is
#: scaled before it is used as a planning number. Deliberately pessimistic:
#: the recommendation should survive being wrong by a factor of several.
PHASE9_VOLUME_SAFETY_FACTOR = 4.0

#: Free capacity must exceed the projected requirement by this factor before
#: a volume is recommended for production rollout bytes.
REQUIRED_HEADROOM_FACTOR = 10.0


def projected_rollout_bytes(total_scheduled_games: int) -> dict:
    """Projected on-disk cost of a number of committed Phase 9 rollout games."""
    if total_scheduled_games < 0:
        raise Phase9StorageError(
            f"scheduled game count must be >= 0, got {total_scheduled_games}"
        )
    base = PHASE8_MEASURED_BYTES_PER_GAME * total_scheduled_games
    projected = base * PHASE9_VOLUME_SAFETY_FACTOR
    return {
        "total_scheduled_games": int(total_scheduled_games),
        "phase8_measured_bytes_per_game": round(PHASE8_MEASURED_BYTES_PER_GAME, 2),
        "phase8_measurement_basis": (
            "352,975,450 bytes over 28,000 committed games of the accepted "
            "synthetic_warmstart_corpus_v1 (shards + metadata + journal)"
        ),
        "safety_factor": PHASE9_VOLUME_SAFETY_FACTOR,
        "base_bytes": int(base),
        "projected_bytes": int(projected),
        "projected_gib": round(projected / 1024**3, 3),
    }


def evaluate_storage_target(path, total_scheduled_games: int) -> dict:
    """Is `path` a defensible production target for a Phase 9 rollout corpus?

    Combines the live volume measurement, a real write probe and the
    projected requirement. `recommended` is true only when the volume is
    mounted, writable, not read-only, and has at least
    :data:`REQUIRED_HEADROOM_FACTOR` times the projected requirement free.
    """
    volume = volume_diagnostics(path)
    writable = check_writable(path)
    projection = projected_rollout_bytes(total_scheduled_games)
    required = projection["projected_bytes"] * REQUIRED_HEADROOM_FACTOR
    headroom = (
        volume["free_bytes"] / projection["projected_bytes"]
        if projection["projected_bytes"]
        else float("inf")
    )
    problems = list(writable["problems"])
    if volume["read_only"]:
        problems.append(f"{volume['mount_point']} is mounted read-only")
    if volume["free_bytes"] < required:
        problems.append(
            f"{volume['mount_point']} has {volume['free_gib']} GiB free, below the "
            f"{REQUIRED_HEADROOM_FACTOR}x headroom requirement of "
            f"{round(required / 1024 ** 3, 3)} GiB"
        )
    return {
        "volume": volume,
        "write_probe": writable,
        "projection": projection,
        "required_headroom_factor": REQUIRED_HEADROOM_FACTOR,
        "required_free_bytes": int(required),
        "observed_headroom_factor": round(headroom, 1),
        "recommended": not problems,
        "problems": problems,
        "identity_rule": STORAGE_IDENTITY_RULE,
    }


__all__ = [
    "DEFAULT_PHASE9_ROLLOUT_ROOT",
    "PHASE8_MEASURED_BYTES_PER_GAME",
    "PHASE9_ROLLOUT_ROOT_ENV",
    "PHASE9_ROLLOUT_ROOT_POINTER",
    "PHASE9_VOLUME_SAFETY_FACTOR",
    "REQUIRED_HEADROOM_FACTOR",
    "STORAGE_IDENTITY_RULE",
    "Phase9StorageError",
    "check_writable",
    "default_rollout_root",
    "describe_rollout_root",
    "evaluate_storage_target",
    "namespace_rollout_directory",
    "projected_rollout_bytes",
    "repository_root",
    "volume_diagnostics",
]
