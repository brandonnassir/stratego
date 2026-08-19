"""Optional Phase 10B: where rollout bytes live, and how that is recorded.

Specification source: `OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`
section 24.

Physical path is diagnostic, never logical identity. Nothing in the schedule,
the seeds, the digests or the commit identities reads a path, so a rollout
copied byte-for-byte to another volume is the same rollout. This module exists
so a relocation is one recorded fact instead of a search-and-replace.

If a pointer names an absent external volume the resolution stops rather than
silently creating an internal replacement, because a half-written run split
across two roots is worse than a refusal.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROLLOUT_ROOT_ENV = "STRATEGO_PHASE10B_ROLLOUT_ROOT"
ROLLOUT_ROOT_POINTER = "data/phase10b_rollout_root.txt"
DEFAULT_ROLLOUT_ROOT = "data/phase10b/rollouts"

IDENTITY_RULE = (
    "a logical Phase 10B identity never contains a path; the resolved root is "
    "recorded as a diagnostic so a relocation is auditable"
)


class Phase10BStorageError(RuntimeError):
    """The Phase 10B rollout root could not be resolved or written to."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rollout_root() -> Path:
    configured = os.environ.get(ROLLOUT_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    pointer = repository_root() / ROLLOUT_ROOT_POINTER
    if pointer.exists():
        recorded = pointer.read_text().strip()
        if recorded:
            return Path(recorded).expanduser()
    return repository_root() / DEFAULT_ROLLOUT_ROOT


def describe() -> dict:
    configured = os.environ.get(ROLLOUT_ROOT_ENV, "").strip()
    pointer = repository_root() / ROLLOUT_ROOT_POINTER
    recorded = pointer.read_text().strip() if pointer.exists() else ""
    if configured:
        source = "environment"
    elif recorded:
        source = "pointer_file"
    else:
        source = "repository_default"
    root = rollout_root()
    total, used, free = (None, None, None)
    anchor = root
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if anchor.exists():
        usage = shutil.disk_usage(anchor)
        total, used, free = usage.total, usage.used, usage.free
    return {
        "root": str(root),
        "source": source,
        "environment_variable": ROLLOUT_ROOT_ENV,
        "environment_value": configured,
        "pointer_file": str(pointer),
        "pointer_value": recorded,
        "repository_default": str(repository_root() / DEFAULT_ROLLOUT_ROOT),
        "resolved_mount_anchor": str(anchor),
        "anchor_exists": anchor.exists(),
        "bytes_total": total,
        "bytes_used": used,
        "bytes_free": free,
        "identity_rule": IDENTITY_RULE,
    }


def resolve_writable(*, required_bytes: int = 0) -> dict:
    """Resolve the root, refuse an absent external volume, and prove writability."""
    description = describe()
    root = Path(description["root"])
    if description["source"] in ("environment", "pointer_file"):
        anchor = Path(description["resolved_mount_anchor"])
        if not anchor.exists():
            raise Phase10BStorageError(
                f"the configured Phase 10B rollout root {root} names an absent "
                f"volume ({anchor}); BLOCKED rather than silently creating an "
                "internal replacement path"
            )
        # A pointer into /Volumes whose mount point vanished resolves to an
        # existing ancestor, which is exactly the silent-fallback case the plan
        # forbids: refuse unless the named volume itself is mounted.
        parts = root.parts
        if len(parts) > 2 and parts[1] == "Volumes":
            mount = Path(parts[0]) / parts[1] / parts[2]
            if not mount.is_mount() and not mount.exists():
                raise Phase10BStorageError(
                    f"external volume {mount} is not mounted; BLOCKED"
                )
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".phase10b_write_probe"
    try:
        probe.write_bytes(b"phase10b")
        probe.unlink()
    except OSError as error:
        raise Phase10BStorageError(f"{root} is not writable: {error}") from error
    usage = shutil.disk_usage(root)
    if required_bytes and usage.free < int(required_bytes):
        raise Phase10BStorageError(
            f"{root} has {usage.free:,} free bytes; the run projects "
            f"{int(required_bytes):,}"
        )
    description.update(
        {
            "writable": True,
            "bytes_free": usage.free,
            "required_bytes": int(required_bytes),
            "headroom_bytes": usage.free - int(required_bytes),
        }
    )
    return description


def projected_bytes(games: int, *, bytes_per_game: int = 40_000) -> int:
    """A deliberately generous projection of one run's rollout bytes.

    The measured Phase 9 canonical store held roughly 40 KB per committed
    game including sidecars and journals; Phase 10B games are the same shape.
    """
    return int(games) * int(bytes_per_game)


__all__ = [
    "DEFAULT_ROLLOUT_ROOT",
    "IDENTITY_RULE",
    "ROLLOUT_ROOT_ENV",
    "ROLLOUT_ROOT_POINTER",
    "Phase10BStorageError",
    "describe",
    "projected_bytes",
    "resolve_writable",
    "rollout_root",
]
