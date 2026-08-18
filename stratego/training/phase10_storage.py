"""Phase 10: where setup-outcome corpus bytes live — and why that is not identity.

Specification sources:

- `00_PHASE_10_SEQUENCE_AND_COMMON_CONTRACT.md` ("Storage/path semantics",
  "Stop conditions")
- `01_AGENT_1_CONTRACT_SEEDS_BANKS_ACCEPTANCE.md` ("Handoff to Agent 2":
  resolver/storage policy)

Why this is a separate module
-----------------------------
:mod:`stratego.training.phase10_schedule` is the logical schedule and
imports nothing from here: it has no path, no environment variable and no
filesystem call anywhere in its derivations. That separation is the
structural proof of the rule this module exists to serve — **a corpus
written at one path and copied byte-for-byte to another is the same
corpus**. Version, logical game ids, payload/metadata digests and commit
identities decide identity; the directory a byte landed in never does.

The resolution order is the accepted Phase 8/9 precedent
(`synthetic_corpus.default_corpus_root`, `phase9_storage.default_rollout_root`),
reused verbatim rather than reinvented:

```text
STRATEGO_PHASE10_CORPUS_ROOT       explicit per-process override
data/phase10_corpus_root.txt       the recorded durable redirect, if any
data/phase10/corpus                the contract's repository default
```

Mount safety
------------
The common contract's stop condition is explicit: *if a pointer names an
absent external volume, stop `BLOCKED`; do not silently create an internal
replacement path*. :func:`check_corpus_root` implements exactly that — a
pointer or environment value naming a `/Volumes/...` mount that is not
currently mounted is a hard `BLOCKED` finding, never a quiet fallback to
the repository default.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: The `phase10_setup_outcome_corpus_v1` root relative to the repository.
DEFAULT_PHASE10_CORPUS_ROOT = "data/phase10/corpus"

#: Per-process override of the corpus root.
PHASE10_CORPUS_ROOT_ENV = "STRATEGO_PHASE10_CORPUS_ROOT"

#: Durable redirect: a one-line file holding the corpus root. Deliberately
#: outside `data/phase10/` so it survives when the corpus bytes do not live
#: there, exactly as `data/phase9_rollout_root.txt` does for Phase 9.
PHASE10_CORPUS_ROOT_POINTER = "data/phase10_corpus_root.txt"

#: The one sentence every consumer of this module has to carry forward.
STORAGE_IDENTITY_RULE = (
    "the resolved corpus root is an operational diagnostic, never an "
    "identity: Phase 10 corpus identity is corpus version + logical game ids "
    "+ payload/metadata digests + commit identities, so the same bytes copied "
    "to another volume are the same corpus"
)

#: Absolute-path prefix of an external macOS mount. A configured root under
#: this prefix must actually be mounted; see :func:`check_corpus_root`.
EXTERNAL_VOLUME_PREFIX = "/Volumes/"


class Phase10StorageError(RuntimeError):
    """Raised when a Phase 10 storage location is unusable as configured."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_corpus_root() -> Path:
    """Where this installation keeps `phase10_setup_outcome_corpus_v1` bytes.

    Every consumer — the collector, the fitter, the auditors — should ask
    this rather than assume a path, so a relocation is one recorded fact
    instead of a search-and-replace. Never called from `phase10_schedule`.
    """
    configured = os.environ.get(PHASE10_CORPUS_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    pointer = repository_root() / PHASE10_CORPUS_ROOT_POINTER
    if pointer.exists():
        recorded = pointer.read_text().strip()
        if recorded:
            return Path(recorded).expanduser()
    return repository_root() / DEFAULT_PHASE10_CORPUS_ROOT


def describe_corpus_root() -> dict:
    """Which redirect (if any) chose the corpus root, for the manifest."""
    configured = os.environ.get(PHASE10_CORPUS_ROOT_ENV, "").strip()
    pointer = repository_root() / PHASE10_CORPUS_ROOT_POINTER
    recorded = pointer.read_text().strip() if pointer.exists() else ""
    if configured:
        source = "environment"
    elif recorded:
        source = "pointer_file"
    else:
        source = "repository_default"
    return {
        "root": str(default_corpus_root()),
        "source": source,
        "environment_variable": PHASE10_CORPUS_ROOT_ENV,
        "environment_value": configured,
        "pointer_file": str(pointer),
        "pointer_value": recorded,
        "repository_default": str(repository_root() / DEFAULT_PHASE10_CORPUS_ROOT),
        "identity_rule": STORAGE_IDENTITY_RULE,
    }


def _volume_mount_point(path: Path) -> "Path | None":
    """The `/Volumes/<name>` mount a path sits under, or `None`."""
    text = str(path)
    if not text.startswith(EXTERNAL_VOLUME_PREFIX):
        return None
    remainder = text[len(EXTERNAL_VOLUME_PREFIX) :].split("/", 1)[0]
    if not remainder:
        return None
    return Path(EXTERNAL_VOLUME_PREFIX + remainder)


def check_corpus_root(path=None) -> dict:
    """Mount-safety and capacity findings for the resolved corpus root.

    A `BLOCKED` finding means exactly what the common contract says it
    means: the configured location names an external volume that is not
    mounted, so the correct response is to stop rather than to invent an
    internal replacement path.
    """
    description = describe_corpus_root()
    target = Path(path) if path is not None else default_corpus_root()
    mount = _volume_mount_point(target)
    blocked: list[str] = []

    if mount is not None and not mount.is_dir():
        blocked.append(
            f"configured corpus root {target} names external volume {mount}, which "
            "is not mounted; stop BLOCKED rather than creating an internal "
            "replacement path"
        )

    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing) if existing.exists() else None

    return {
        "resolved_root": str(target),
        "description": description,
        "external_volume": str(mount) if mount is not None else None,
        "external_volume_mounted": None if mount is None else bool(mount.is_dir()),
        "nearest_existing_ancestor": str(existing),
        "free_bytes": None if usage is None else int(usage.free),
        "total_bytes": None if usage is None else int(usage.total),
        "blocked": blocked,
        "usable": not blocked,
    }


def storage_policy_document() -> dict:
    """The machine-readable storage policy handed to Agent 2."""
    return {
        "corpus_version": "phase10_setup_outcome_corpus_v1",
        "resolution_order": [
            PHASE10_CORPUS_ROOT_ENV,
            PHASE10_CORPUS_ROOT_POINTER,
            DEFAULT_PHASE10_CORPUS_ROOT,
        ],
        "identity_rule": STORAGE_IDENTITY_RULE,
        "logical_identity_is_path_independent": True,
        "hard_coded_volume_paths_forbidden": (
            "no logical scheduling, model or selector identity may contain a "
            "/Volumes/... path; the resolved root appears only in manifests as a "
            "diagnostic"
        ),
        "absent_external_volume_rule": (
            "a configured root naming an unmounted /Volumes/... volume is BLOCKED; "
            "never silently create an internal replacement path"
        ),
        "preferred_location": (
            "the verified external volume for substantial corpus/replay bytes, "
            "recorded through the pointer file"
        ),
    }


__all__ = [
    "DEFAULT_PHASE10_CORPUS_ROOT",
    "EXTERNAL_VOLUME_PREFIX",
    "PHASE10_CORPUS_ROOT_ENV",
    "PHASE10_CORPUS_ROOT_POINTER",
    "STORAGE_IDENTITY_RULE",
    "Phase10StorageError",
    "check_corpus_root",
    "default_corpus_root",
    "describe_corpus_root",
    "repository_root",
    "storage_policy_document",
]
