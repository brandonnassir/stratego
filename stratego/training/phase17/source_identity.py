"""Phase 17 Agent 4C: the production source closure and its digest.

Specification source: Agent 4C instruction section 1.1.

Why a source digest and not just a commit
------------------------------------------
A commit says what the repository was asked to be. A source closure says what
the process actually loaded. The two differ exactly when it matters: an edited
working tree, a stale `__pycache__`, a file restored from another branch. The
12-hour run publishes 25 immutable candidates whose only claim to comparability
is that they came from one program, so that claim is recomputed from the bytes
on disk at start, bound into the run digest, and re-checked on every resume.

The closure is enumerated, not listed
--------------------------------------
`PRODUCTION_SOURCE_ROOTS` names *directories and files*, and the closure is the
sorted expansion of them. The pre-correction closure was a hand-written tuple
of eleven paths that omitted the whole move half -- `move_trainer.py`,
`transition_collector.py`, `transition_targets.py` and the rest -- so a change
to the move learner did not move the digest. A hand-maintained list of a
package's files is a list that goes stale the first time somebody adds a
module; expanding the package is the version that cannot.

`scripts/run_phase17_d10_smoke.py` is deliberately NOT in the closure. It is
not loaded by the production process, and a source identity that moves when a
smoke script is edited would force Agent 6 to refreeze the digest for a change
that cannot reach the run.

Nothing here is hard-coded
--------------------------
There is no frozen digest constant in this module on purpose. Agent 6 freezes
the recomputed value into the launch manifest and Agent 7 passes it back on the
production command, so the check is "the bytes I just hashed are the bytes the
operator authorized" rather than "the bytes I just hashed are the bytes that
were here when this file was written".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .move_contract import Phase17MoveError

SOURCE_CLOSURE_VERSION = "phase17_production_source_closure_v1"

#: What the production process loads. Directories expand to their sorted
#: `*.py`; files are taken as they are.
PRODUCTION_SOURCE_ROOTS = (
    "stratego/training/phase17",
    "scripts/run_phase17_training.py",
)


class Phase17SourceError(Phase17MoveError):
    """The production source closure is missing, empty, or not the frozen one."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def repository_root() -> Path:
    """The repository this module was loaded from."""
    return Path(__file__).resolve().parents[3]


def expand_roots(roots=PRODUCTION_SOURCE_ROOTS, *, root: "str | Path | None" = None) -> list:
    """Every closure member, repository-relative, sorted and de-duplicated."""
    base = Path(root) if root is not None else repository_root()
    names: set = set()
    for entry in roots:
        target = base / entry
        if target.is_dir():
            for candidate in target.rglob("*.py"):
                if "__pycache__" in candidate.parts:
                    continue
                names.add(candidate.relative_to(base).as_posix())
        elif target.is_file():
            names.add(target.relative_to(base).as_posix())
        else:
            raise Phase17SourceError(
                f"the production source closure names {entry!r}, which is not a "
                f"file or directory under {base}"
            )
    if not names:
        raise Phase17SourceError(
            f"the production source closure expanded to nothing under {base}"
        )
    return sorted(names)


def source_closure(
    roots=PRODUCTION_SOURCE_ROOTS, *, root: "str | Path | None" = None
) -> dict:
    """The closure document: every member path, its sha256, and their digest."""
    base = Path(root) if root is not None else repository_root()
    entries = [
        {"path": name, "sha256": _file_sha256(base / name)}
        for name in expand_roots(roots, root=base)
    ]
    return {
        "closure_version": SOURCE_CLOSURE_VERSION,
        "roots": list(roots),
        "file_count": len(entries),
        "files": entries,
        "closure_digest": _json_digest(entries),
    }


def production_source_closure(*, root: "str | Path | None" = None) -> dict:
    """The closure of exactly what `run_phase17_training.py --start` executes."""
    return source_closure(PRODUCTION_SOURCE_ROOTS, root=root)


def require_source_digest(value, *, context: str = "this process") -> str:
    """A source digest that could actually identify a program, or a refusal.

    Empty, `None`, whitespace and anything that is not a sha256 hex string are
    all refused with the same message. An empty digest is the dangerous case:
    it compares equal to the empty digest in every checkpoint and export, so a
    run that never bound its source would pass every identity check it has.
    """
    if value is None or not isinstance(value, str) or not value.strip():
        raise Phase17SourceError(
            f"{context} has no production source digest. Every checkpoint, "
            "export and resume binds it, and an empty digest matches every "
            "other empty digest, so the run could not prove which program "
            "produced its candidates. Pass --source-digest with the value "
            "Agent 6 froze in the launch manifest."
        )
    digest = value.strip()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise Phase17SourceError(
            f"{context} was given source digest {value!r}, which is not a "
            "sha256 hex digest"
        )
    return digest


def verify_source_digest(
    expected: str, *, root: "str | Path | None" = None, context: str = "this process"
) -> dict:
    """Recompute the closure and refuse anything but an exact match.

    Returns the closure document so the caller can record what it hashed, not
    merely that it agreed.
    """
    wanted = require_source_digest(expected, context=context)
    closure = production_source_closure(root=root)
    if closure["closure_digest"] != wanted:
        raise Phase17SourceError(
            f"{context}: the working tree's Phase 17 production closure digests "
            f"to {closure['closure_digest']} over {closure['file_count']} files, "
            f"not the authorized {wanted}. The code on disk is not the code the "
            "launch manifest froze; refusing to start."
        )
    return closure


__all__ = [
    "PRODUCTION_SOURCE_ROOTS",
    "Phase17SourceError",
    "SOURCE_CLOSURE_VERSION",
    "expand_roots",
    "production_source_closure",
    "repository_root",
    "require_source_digest",
    "source_closure",
    "verify_source_digest",
]
