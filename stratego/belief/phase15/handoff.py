"""Phase 15 Agent 1 section 13: the search handoff.

Specification source: `01_AGENT_1_BELIEF_HEAD_TRAINING.md` section 13.

> The handoff must bind exact digests for P18, P24, B18, B24, the corpus,
> calibration values, provider interface version, and accepted sampler
> version.

One document, every identity
----------------------------
:func:`build_handoff` is the single place those bindings are assembled, and
:func:`verify_handoff` re-derives every digest it names from the bytes on
disk. A later agent that wants to know whether the handoff still describes
reality runs the second function; it does not read the first.

What the handoff deliberately does not say
-------------------------------------------
It does not name a final player, a search configuration, a budget or a
strength claim. Section 14: "Do not implement search, choose the final
combined player, modify Phase 12, or control any running Phase 14 task."
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ...training.phase11_contract import BELIEF_SAMPLER_VERSION
from .contract import (
    CORPUS_VERSION,
    PHASE15_STATUS_MARKERS,
    SPECIALISTS,
    SPECIALIST_SOURCE,
    Phase15Error,
)
from .heads import BELIEF_ARCHITECTURE_VERSION
from .interface import BELIEF_INTERFACE_VERSION
from .metrics import METRICS_VERSION

#: The handoff identity.
HANDOFF_VERSION = "phase15_search_handoff_v1"


class Phase15HandoffError(Phase15Error):
    """The handoff could not be built or no longer describes reality."""


def _sha256(path: "Path | str", *, chunk: int = 1 << 20) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def repository_relative(path: "Path | str", root: "Path | str" = ".") -> str:
    """A repo-relative path where possible, so the handoff travels.

    Paths outside the repository — a Phase 14 archive on an external volume —
    are left absolute, because that is genuinely where they are.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def build_handoff(
    *,
    sources: dict,
    specialists: dict,
    corpus_manifest: dict,
    interface_reports: dict,
    development_metrics: dict,
    root: "Path | str" = ".",
) -> dict:
    """Assemble the handoff document.

    `sources` maps `p18`/`p24` to a source identity block, `specialists`
    maps `b18`/`b24` to the checkpoint identity returned by
    :func:`~.checkpoint.save_specialist` plus its calibration record.
    """
    for source_id, block in sources.items():
        if not block.get("model_state_digest"):
            raise Phase15HandoffError(f"{source_id} has no model-state digest")
    for specialist_id in SPECIALISTS:
        if specialist_id not in specialists:
            raise Phase15HandoffError(f"the handoff is missing {specialist_id}")

    return {
        "artifact": HANDOFF_VERSION,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **PHASE15_STATUS_MARKERS,
        "policy_models": {
            source_id: {
                "logical_identity": block["logical_identity"],
                "phase14_candidate_hour": block["hour"],
                "model_state_digest": block["model_state_digest"],
                "checkpoint_path": repository_relative(block["phase15_copy_path"], root),
                "checkpoint_sha256": block["phase15_copy_sha256"],
                "phase14_archive_path": block["original_snapshot_path"],
                "phase14_archive_sha256": block["original_snapshot_sha256"],
                "global_optimizer_step": block["global_optimizer_step"],
                "role": "policy and value. Immutable. Phase 15 never trained it.",
            }
            for source_id, block in sorted(sources.items())
        },
        "belief_models": {
            specialist_id: {
                "bound_policy": SPECIALIST_SOURCE[specialist_id],
                "checkpoint_path": repository_relative(block["path"], root),
                "checkpoint_sha256": block["sha256"],
                "state_digest": block["state_digest"],
                "architecture_version": BELIEF_ARCHITECTURE_VERSION,
                "calibration": {
                    "applied_temperature": block["temperature"],
                    "fitted_temperature": block["calibration"]["temperature"],
                    "keep_calibrated": block["calibration"]["keep_calibrated"],
                    "fitted_on": "calibration split only",
                    "top1_labels_changed": block["calibration"]["top1_labels_changed"],
                    "note": (
                        "section 10 keeps a temperature only if it improves both "
                        "development NLL and calibration error; when it does not, "
                        "the applied temperature is 1.0 and the model is used raw"
                    ),
                },
                "role": (
                    "hidden-rank marginals only. Holds no policy or value "
                    "parameter; its outputs feed the accepted sampler."
                ),
            }
            for specialist_id, block in sorted(specialists.items())
        },
        "corpus": {
            "corpus_version": CORPUS_VERSION,
            "corpus_digest": corpus_manifest["corpus_digest"],
            "corpus_format_version": corpus_manifest["corpus_format_version"],
            "run_version": corpus_manifest["run_version"],
            "positions": {
                split: block["samples"]
                for split, block in sorted(corpus_manifest["splits"].items())
            },
            "orientation_rule": corpus_manifest["orientation"]["orientation_rule"],
            "supersedes": (
                "phase11b_common_corpus_v1 and the Phase 12 match packs, both "
                "contaminated by the Blue setup-orientation defect"
            ),
        },
        "provider": {
            "interface_version": BELIEF_INTERFACE_VERSION,
            "entry_points": [
                "predict_marginals(public_state) -> {piece_slot: 12-way probabilities}",
                "sample_worlds(public_state, n, seed) -> complete legal hidden armies",
                "sample_assignments(public_state, n, seed) -> [{piece_slot: rank}]",
            ],
            "public_state_type": (
                "stratego.belief.phase11b.interface.Phase11BPublicState, reused by "
                "import; the accepted phase11_public_state_v1 document plus the "
                "127-channel observation, and no field a true rank could arrive in"
            ),
            "accepted_sampler_version": BELIEF_SAMPLER_VERSION,
            "accepted_sampler_source": (
                "stratego.evaluation.phase11_sampler.sample_belief_world, imported "
                "and unmodified; marginals are not sampled per piece and no "
                "inventory or movement-impossibility constraint was altered"
            ),
            "checks": interface_reports,
        },
        "metrics": {
            "metrics_version": METRICS_VERSION,
            "development": development_metrics,
            "note": (
                "belief-quality metrics on the Phase 15 development split. Not a "
                "playing-strength claim: no search was implemented or evaluated."
            ),
        },
        "not_included": [
            "any search implementation",
            "any choice of final combined player",
            "any modification to Phase 12",
            "any control of a Phase 14 task",
        ],
    }


def verify_handoff(document: dict, *, root: "Path | str" = ".") -> dict:
    """Re-derive every digest the handoff names, from the bytes on disk."""
    root = Path(root)
    findings: list[str] = []
    checked = 0
    for group, key in (("policy_models", "checkpoint"), ("belief_models", "checkpoint")):
        for name, block in document[group].items():
            path = Path(block[f"{key}_path"])
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                findings.append(f"{group}.{name}: {path} is missing")
                continue
            observed = _sha256(path)
            if observed != block[f"{key}_sha256"]:
                findings.append(
                    f"{group}.{name}: {path.name} is {observed[:16]}, the handoff "
                    f"records {block[f'{key}_sha256'][:16]}"
                )
            checked += 1
    return {
        "artifacts_checked": checked,
        "findings": findings,
        "verified": not findings,
    }


def write_handoff(document: dict, path: "Path | str") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "HANDOFF_VERSION",
    "Phase15HandoffError",
    "build_handoff",
    "repository_relative",
    "verify_handoff",
    "write_handoff",
]
