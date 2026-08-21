"""Phase 14: the immutable launch package and the durable control files.

Specification source: `04_AGENT_4_IMMUTABLE_LAUNCH_PACKAGE.md` sections 1, 4,
5 and 17.

Why digests of the config are not enough
----------------------------------------
The Phase 13 Agent 3 loader-pool repair changed neither the contract digest
(``62ce6d4e…``) nor the integrated config digest (``9c2a38e4…``): both are
computed over frozen *values* and module version strings, not over the bytes
that implement them. That is the right design for a configuration identity and
exactly the wrong thing to rely on for "is the fix installed". A run launched
against the pre-repair code would present both correct digests and would still
die the first time a CPU loader worker was killed.

So the launch manifest binds the code as well: the committed Git revision, the
tracked working-tree state, and a content digest over every ``stratego``
module the Phase 14 training graph actually imports — 111 files, including the
two that carry the repair. :func:`assert_launch_code` recomputes all of it at
launch and refuses to start on anything else. Rebuilding the manifest is a
deliberate act (``scripts/phase14_build_launch_package.py``), which is the
point: the code may change, but not silently and not between hour 0 and hour
168.

The durable control files
-------------------------
A killed process cannot record its own death, and an in-process control
surface cannot be reached from outside the process. Both facts push the two
operator controls onto disk, under the external run directory:

``phase14_emergency_stop.json``
    Written by the operator. The learner stops at its next safe boundary and
    the supervisor refuses to restart it.
``phase14_integrity_failure.json``
    Written by the learner when it dies of an unrecoverable integrity failure.
    The supervisor refuses to restart over one.

Neither file changes a training value, and neither can: the frozen keys are
refused by name by the accepted control surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from .phase14_contract import PHASE14_NAMESPACE, repository_root

PHASE14_LAUNCH_VERSION = "phase14_launch_v1"
LAUNCH_MANIFEST_ARTIFACT = "phase14_launch_manifest_v1"
FINAL_CONFIG_ARTIFACT = "phase14_final_training_config_v1"

LAUNCH_MANIFEST_RELATIVE_PATH = "reports/phase13/phase14_launch_manifest_v1.json"
FINAL_CONFIG_RELATIVE_PATH = "reports/phase13/phase14_final_training_config_v1.json"

#: The committed revision that carries the accepted Agent 3 worker-pool
#: repair. Recorded as provenance: the manifest binds the code by content, but
#: an operator reading the manifest should be able to see which commit the
#: accepted repair arrived in without running `git log`.
AGENT3_REPAIR_REVISION = "e6daae8df7e1da697263635db0aadc70651b3dd8"

#: Entry points whose import closure *is* the Phase 14 training system. Listed
#: explicitly because several of the runner's imports are lazy (inside methods),
#: and a closure that missed them would bind less code than the run executes.
ENTRY_POINT_MODULES = (
    "stratego.training.phase14_runner",
    "stratego.training.phase14_trainer",
    "stratego.training.phase14_collector",
    "stratego.training.phase14_checkpoint",
    "stratego.training.phase14_pool",
    "stratego.training.phase14_schedule",
    "stratego.training.phase14_seed",
    "stratego.training.phase14_setup_source",
    "stratego.training.phase14_storage",
    "stratego.training.phase14_telemetry",
    "stratego.training.phase14_clock",
    "stratego.training.phase14_contract",
    "stratego.training.phase14_config",
    "stratego.training.phase14_status",
    "stratego.training.phase14_launch",
    "stratego.training.phase14_supervisor",
    "stratego.evaluation.phase14_candidates",
)

#: The operator-facing scripts the manifest binds by content. A launch that
#: used a different launcher would be a different launch.
DEFAULT_SCRIPTS = {
    "launch": "scripts/phase14_launch.py",
    "resume": "scripts/phase14_launch.py",
    "status": "scripts/phase14_status.py",
    "emergency_stop": "scripts/phase14_emergency_stop.py",
    "candidate_evaluator": "scripts/phase14_evaluate_candidates.py",
    "final_selection": "scripts/phase14_select_final.py",
    "rebuild_launch_package": "scripts/phase14_build_launch_package.py",
}

EMERGENCY_STOP_FILENAME = "phase14_emergency_stop.json"
INTEGRITY_FAILURE_FILENAME = "phase14_integrity_failure.json"

#: The operational topology Agent 3 rehearsed at the frozen production
#: population and Agent 4 freezes. Deliberately *not* part of the logical
#: config digest — Agent 2 excluded operational choices on purpose, because
#: two identical experiments on different machines are the same experiment.
FROZEN_TOPOLOGY = {
    "device": "mps",
    "inference_device": "mps",
    "loader_workers": 6,
    "games_in_flight": 96,
    "inference_batch_shape": 64,
    "population": "production",
    "games_per_iteration": 2048,
    "source": "Phase 13 Agent 3 rehearsal, run at this exact topology for 90 minutes",
}


class Phase14LaunchError(RuntimeError):
    """Raised when Phase 14 may not launch under the bound launch package."""


def _utc_text(unix: "float | None" = None) -> str:
    from .phase14_status import utc_text

    return utc_text(unix)


# ---------------------------------------------------------------------------
# Binding the code
# ---------------------------------------------------------------------------


def _git(*arguments: str) -> "str | None":
    try:
        finished = subprocess.run(
            ["git", *arguments],
            cwd=str(repository_root()),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.rstrip("\n")


def git_state() -> dict:
    """HEAD, the branch, and every *tracked* file that differs from it.

    Untracked files are excluded deliberately: a scratch file beside the
    package is not a change to the code the run executes, and a launch check
    that trips on one would be turned off by the second week.
    """
    revision = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain", "--untracked-files=no")
    dirty = sorted(
        line[3:].strip() for line in (porcelain or "").splitlines() if line.strip()
    )
    return {
        "revision": revision,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked_files": dirty,
        "tracked_tree_clean": not dirty,
        "agent3_repair_revision": AGENT3_REPAIR_REVISION,
    }


#: The program that computes the closure. Run in a *fresh* interpreter, and
#: that is the whole point: `sys.modules` in the calling process carries
#: whatever else it happened to import, so computing the closure in place would
#: make the bound file set depend on the caller. It was caught exactly that
#: way — a full test-suite run had already imported `stratego.search`, which
#: Phase 14 training never touches, and the closure silently grew.
_CLOSURE_PROGRAM = """
import json, sys, importlib
from pathlib import Path
for name in {entry_points!r}:
    importlib.import_module(name)
root = Path({root!r})
files = set()
for name, module in list(sys.modules.items()):
    if not name.startswith("stratego"):
        continue
    path = getattr(module, "__file__", None)
    if not path:
        continue
    try:
        files.add(str(Path(path).resolve().relative_to(root)))
    except ValueError:
        continue
print(json.dumps(sorted(files)))
"""


@lru_cache(maxsize=1)
def code_closure() -> tuple:
    """Every `stratego` source file the Phase 14 training graph imports.

    The declared entry points are imported in a clean subprocess and the
    closure is read from its `sys.modules`, so a module reached only through a
    lazy import inside a method is bound, and a module the *caller* happened to
    import is not.
    """
    root = repository_root()
    finished = subprocess.run(
        [
            sys.executable,
            "-c",
            _CLOSURE_PROGRAM.format(
                entry_points=list(ENTRY_POINT_MODULES), root=str(root)
            ),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    if finished.returncode != 0:
        raise Phase14LaunchError(
            "the Phase 14 code closure could not be computed; a launch may not be "
            f"bound to a file set that is not reproducible:\n{finished.stderr[-2000:]}"
        )
    return tuple(json.loads(finished.stdout))


def file_sha256(path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def code_file_digests() -> dict:
    root = repository_root()
    return {relative: file_sha256(root / relative) for relative in code_closure()}


def code_digest(digests: "dict | None" = None) -> str:
    """One content digest over the whole Phase 14 training import closure."""
    digests = code_file_digests() if digests is None else digests
    hasher = hashlib.sha256()
    hasher.update(PHASE14_LAUNCH_VERSION.encode())
    for relative in sorted(digests):
        hasher.update(f"{relative}|{digests[relative]}\n".encode())
    return hasher.hexdigest()


def worker_repair_evidence() -> dict:
    """Positive proof the accepted Agent 3 worker-pool repair is installed.

    Digests prove "the same code as when the manifest was built". These
    assertions prove "the code contains the repair" — a different question, and
    the one the task asks, because both frozen digests are blind to it.
    """
    import inspect
    from concurrent.futures import BrokenExecutor

    from .phase14_runner import RECOVERABLE_ERRORS
    from .phase14_trainer import (
        MAX_LOADER_POOL_REBUILDS,
        Phase14Trainer,
    )

    source = inspect.getsource(Phase14Trainer._next_minibatch)
    checks = {
        "broken_executor_is_recoverable": BrokenExecutor in RECOVERABLE_ERRORS,
        "next_minibatch_catches_broken_executor": "except BrokenExecutor" in source,
        "next_minibatch_rebuilds_at_same_cursor": "rebuilt.next(self.cursor)" in source,
        "rebuilds_are_counted": "loader_pool_rebuilds" in source,
        "rebuild_cap_present": MAX_LOADER_POOL_REBUILDS == 16,
        "rebuild_events_recorded": hasattr(Phase14Trainer, "loader_pool_state"),
    }
    return {
        "installed": all(checks.values()),
        "max_loader_pool_rebuilds": int(MAX_LOADER_POOL_REBUILDS),
        "recoverable_errors": [error.__name__ for error in RECOVERABLE_ERRORS],
        "checks": checks,
        "why_digests_do_not_prove_this": (
            "the contract and integrated-config digests are computed over frozen "
            "values and module version strings, both of which the repair left "
            "unchanged"
        ),
    }


def code_binding() -> dict:
    """The complete code identity the launch manifest binds."""
    digests = code_file_digests()
    return {
        "git": git_state(),
        "python": sys.version.split()[0],
        "closure_entry_points": list(ENTRY_POINT_MODULES),
        "closure_files": len(digests),
        "code_digest": code_digest(digests),
        "file_sha256": digests,
        "worker_pool_repair": worker_repair_evidence(),
        # Section 3 of the launch task, answered structurally rather than by
        # assertion: the closure is every module the training graph imports, and
        # no search module is in it.
        "search_modules_in_training_closure": [
            name for name in digests if name.startswith("stratego/search/")
        ],
        "search_excluded": not any(
            name.startswith("stratego/search/") for name in digests
        ),
    }


def assert_launch_code(manifest: dict) -> dict:
    """Refuse to launch on anything but the bound code revision.

    Three independent failures are caught here, and each is a different way the
    real run could stop being the run the manifest describes:

    * a *different commit* — someone launched from another branch or after a
      later commit landed;
    * *the same commit with edited files* — the digest map disagrees even
      though `git rev-parse` does not;
    * *the repair missing* — the strongest check, and the only one that would
      catch a revert that happened to restore an older manifest as well.
    """
    expected = manifest.get("code", {})
    observed = code_binding()
    problems = []

    expected_revision = (expected.get("git") or {}).get("revision")
    observed_revision = observed["git"]["revision"]
    if expected_revision and observed_revision != expected_revision:
        problems.append(
            f"code revision {observed_revision} is not the bound "
            f"{expected_revision}; rebuild the launch package deliberately "
            "before Phase 14 begins"
        )

    expected_dirty = list((expected.get("git") or {}).get("dirty_tracked_files", []))
    observed_dirty = observed["git"]["dirty_tracked_files"]
    if observed_dirty != expected_dirty:
        problems.append(
            f"tracked working-tree state differs from the manifest: bound "
            f"{expected_dirty or 'clean'}, observed {observed_dirty or 'clean'}"
        )

    if expected.get("code_digest") and observed["code_digest"] != expected["code_digest"]:
        differing = sorted(
            relative
            for relative, digest in observed["file_sha256"].items()
            if expected.get("file_sha256", {}).get(relative) != digest
        )
        missing = sorted(
            set(expected.get("file_sha256", {})) - set(observed["file_sha256"])
        )
        problems.append(
            f"the Phase 14 code closure does not match the manifest: "
            f"{len(differing)} file(s) differ ({differing[:5]}), {len(missing)} bound "
            f"file(s) absent ({missing[:5]})"
        )

    for name, entry in sorted((manifest.get("scripts") or {}).items()):
        relative = entry.get("path")
        expected_digest = entry.get("sha256")
        if not relative or not expected_digest:
            continue
        target = repository_root() / relative
        if not target.exists():
            problems.append(f"the bound {name} script is missing at {relative}")
            continue
        observed_digest = file_sha256(target)
        if observed_digest != expected_digest:
            problems.append(
                f"the {name} script {relative} has SHA-256 {observed_digest}, not the "
                f"bound {expected_digest}"
            )

    if not observed["worker_pool_repair"]["installed"]:
        failed = [
            name
            for name, ok in observed["worker_pool_repair"]["checks"].items()
            if not ok
        ]
        problems.append(
            f"the accepted Agent 3 worker-pool repair is not installed: {failed}"
        )

    if problems:
        raise Phase14LaunchError(
            "Phase 14 may not launch on this code revision:\n  - "
            + "\n  - ".join(problems)
        )
    return {
        "verified": True,
        "revision": observed_revision,
        "code_digest": observed["code_digest"],
        "closure_files": observed["closure_files"],
        "worker_pool_repair_installed": True,
    }


def launch_manifest_path() -> Path:
    return repository_root() / LAUNCH_MANIFEST_RELATIVE_PATH


def load_launch_manifest(path=None) -> dict:
    target = Path(path) if path is not None else launch_manifest_path()
    if not target.exists():
        raise Phase14LaunchError(
            f"the Phase 14 launch manifest is missing at {target}; Phase 14 launches "
            "from a bound manifest or not at all"
        )
    return json.loads(target.read_text())


def assert_bound_launch_code(path=None) -> dict:
    """Load the bound manifest and refuse anything but its code revision."""
    return assert_launch_code(load_launch_manifest(path))


# ---------------------------------------------------------------------------
# Operational topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperationalTopology:
    """How the run is executed, as opposed to what the run *is*.

    Frozen from what Agent 3 actually rehearsed for 90 minutes at the frozen
    2,048-game production population. Recorded in the launch manifest and
    deliberately kept out of the logical config digest.
    """

    device: str = "mps"
    inference_device: str = "mps"
    loader_workers: int = 6
    games_in_flight: int = 96
    inference_batch_shape: int = 64

    @staticmethod
    def frozen() -> "OperationalTopology":
        return OperationalTopology(
            device=FROZEN_TOPOLOGY["device"],
            inference_device=FROZEN_TOPOLOGY["inference_device"],
            loader_workers=int(FROZEN_TOPOLOGY["loader_workers"]),
            games_in_flight=int(FROZEN_TOPOLOGY["games_in_flight"]),
            inference_batch_shape=int(FROZEN_TOPOLOGY["inference_batch_shape"]),
        )

    @staticmethod
    def from_manifest(manifest: dict) -> "OperationalTopology":
        block = dict(manifest.get("operational_topology") or {})
        frozen = OperationalTopology.frozen()
        return OperationalTopology(
            device=str(block.get("device", frozen.device)),
            inference_device=str(block.get("inference_device", frozen.inference_device)),
            loader_workers=int(block.get("loader_workers", frozen.loader_workers)),
            games_in_flight=int(block.get("games_in_flight", frozen.games_in_flight)),
            inference_batch_shape=int(
                block.get("inference_batch_shape", frozen.inference_batch_shape)
            ),
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "population": FROZEN_TOPOLOGY["population"],
            "games_per_iteration": FROZEN_TOPOLOGY["games_per_iteration"],
            "source": FROZEN_TOPOLOGY["source"],
            "in_logical_config_digest": False,
        }


def assert_frozen_topology(topology: OperationalTopology) -> OperationalTopology:
    """Refuse a production launch on a topology nobody rehearsed."""
    frozen = OperationalTopology.frozen()
    if topology != frozen:
        raise Phase14LaunchError(
            f"the production topology is frozen at {frozen.to_dict()}; this launch "
            f"declared {topology.to_dict()}. Section 5 forbids benchmarking "
            "alternative worker counts now; rebuild the launch package if a genuine "
            "launch incompatibility is discovered"
        )
    return topology


# ---------------------------------------------------------------------------
# The durable control files
# ---------------------------------------------------------------------------


def emergency_stop_path(external_root) -> Path:
    return Path(external_root) / EMERGENCY_STOP_FILENAME


def request_emergency_stop(external_root, reason: str = "operator request") -> dict:
    """Ask the run to stop at its next safe boundary, from another process.

    A *request*, not a kill: the learner finishes the collection unit or the
    optimizer step in flight and writes a hot checkpoint, because a torn
    iteration is a thing the store then has to reconcile. The supervisor reads
    the same file and does not restart over it.
    """
    path = emergency_stop_path(external_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "artifact": "phase14_emergency_stop_v1",
        "requested_utc": _utc_text(),
        "requested_unix": __import__("time").time(),
        "reason": str(reason),
        "requested_by_pid": os.getpid(),
        "effect": "the learner stops at the next safe boundary; the supervisor does not restart it",
        "does_not_change": "no frozen training value; the deadline is untouched",
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def clear_emergency_stop(external_root) -> dict:
    path = emergency_stop_path(external_root)
    existed = path.exists()
    if existed:
        path.unlink()
    return {"cleared": existed, "path": str(path)}


def emergency_stop_state(external_root) -> dict:
    path = emergency_stop_path(external_root)
    if not path.exists():
        return {"active": False, "path": str(path)}
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # An unreadable stop file is still a stop file. Failing open here would
        # restart a run the operator asked to stop.
        record = {"reason": "unreadable emergency-stop file"}
    return {"active": True, "path": str(path), **record}


def integrity_failure_path(external_root) -> Path:
    return Path(external_root) / INTEGRITY_FAILURE_FILENAME


def record_integrity_failure(external_root, *, error: str, traceback_text: str = "") -> dict:
    """Persist an unrecoverable integrity failure so a restart cannot ignore it."""
    path = integrity_failure_path(external_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "artifact": "phase14_integrity_failure_v1",
        "recorded_utc": _utc_text(),
        "pid": os.getpid(),
        "error": str(error)[:4000],
        "traceback": str(traceback_text)[:8000],
        "effect": "the supervisor refuses to restart; the run needs a human",
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def integrity_failure_state(external_root) -> dict:
    path = integrity_failure_path(external_root)
    if not path.exists():
        return {"recorded": False, "path": str(path)}
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        record = {"error": "unreadable integrity-failure file"}
    return {"recorded": True, "path": str(path), **record}


# ---------------------------------------------------------------------------
# The immutable launch package
# ---------------------------------------------------------------------------


def final_training_config_document() -> dict:
    """`phase14_final_training_config_v1`: everything the run *is*.

    Built from the live modules rather than transcribed, so a value that
    drifted in code could not sit here looking correct. Section 16 of the task
    lists the minimum bindings; each appears below under its own name.
    """
    from .phase14_config import integrated_config_document, integrated_config_digest
    from .phase14_contract import (
        AGENT1C_CHECKPOINT,
        AGENT1C_ROLE,
        EMA_STATE_RECORD,
        FROZEN_METRIC_LIST,
        IMMUTABLE_CONTROL_KEYS,
        OPTIMIZER_STATE_AT_START,
        SELECTION_RULE,
        contract_digest,
    )
    from .phase14_status import status_semantics
    from .phase14_supervisor import supervisor_semantics
    from .phase14_telemetry import EXTENDED_METRIC_PATHS

    integrated = integrated_config_document()
    document = {
        "artifact": FINAL_CONFIG_ARTIFACT,
        "phase": 13,
        "agent": 4,
        "status": "FROZEN",
        "purpose": (
            "the immutable configuration the 168-hour Phase 14 final training run "
            "executes. Building this document starts nothing."
        ),
        "upstream": {
            "agent1_contract": "phase13_final_training_contract_v1",
            "agent2_integrated_config": "phase13_integrated_training_config_v1",
            "agent3_rehearsal": "phase13_rehearsal_v1",
            "phase14_contract_digest": contract_digest(),
            "integrated_config_digest": integrated_config_digest(),
        },
        # Sections 4 and 16: the starting model and the roles around it.
        "starting_model": {
            **integrated["starting_checkpoint"],
            "role": "Phase 14 starting policy/value AND belief auxiliary head",
            "optimizer_state": OPTIMIZER_STATE_AT_START,
            "ema": EMA_STATE_RECORD,
            "agent1c": {
                "checkpoint": AGENT1C_CHECKPOINT,
                "role": AGENT1C_ROLE,
                "used_as_phase14_policy_value": False,
            },
        },
        "optimizer_family": integrated["training_objective"]["optimizer"],
        "objectives": {
            "policy": "PPO clipped surrogate over advantage-filtered learner decisions",
            "value": "categorical cross-entropy over the 3-class W/D/L head, weight 0.5",
            "belief_auxiliary": "accepted Phase 9 belief targets, weight 0.25",
            "behavior_kl": "adaptive beta against the frozen behavior policy",
            "entropy_coefficient": integrated["entropy_coefficient"],
            "detail": integrated["training_objective"],
        },
        "learning_rate": integrated["learning_rate"],
        "transition": integrated["transition"],
        "opponent_mixture": integrated["opponent_mixture"],
        "historical": integrated["historical_pool"],
        "setup_source": integrated["setup_source"],
        "checkpoint_cadences": integrated["checkpoint_cadences"],
        "candidate_evaluation": {
            **integrated["candidate_evaluation"],
            "selection_rule_identity": SELECTION_RULE,
            "search_permitted": False,
            "runs": "out of band, in its own process, never inside the training loop",
            "failure_policy": (
                "preserve the candidate, record the failure, continue training, retry "
                "later on the identical pack"
            ),
            "hour_168_gate": (
                "every marked candidate must carry a complete 128-game result before "
                "the frozen selection rule is applied"
            ),
        },
        "storage_policy": integrated["storage_policy"],
        "deadline_semantics": {
            **integrated["deadline_semantics"],
            "derivation": (
                "run_deadline_utc = run_start_utc + 604800s, materialized exactly once "
                "at launch and persisted in every hot checkpoint"
            ),
            "downtime": "counts against the deadline",
            "post_deadline": "recovery finalizes and takes zero optimizer steps",
        },
        # Section 7 of the Agent 4 additions: the honest checkpoint-age story.
        "recovery_semantics": recovery_semantics(),
        # Section 8 of the Agent 4 additions.
        "rng_semantics": rng_semantics(),
        "monitoring": {
            "frozen_metrics": list(FROZEN_METRIC_LIST),
            "extended_metrics": list(EXTENDED_METRIC_PATHS),
            "immutable_control_keys": list(IMMUTABLE_CONTROL_KEYS),
            "committed_games": status_semantics()["committed_games"],
            "process_counter_games": status_semantics()["process_counter_games"],
            "worker_health": status_semantics()["worker_health"],
            "controls": "emergency stop only; every frozen key is refused by name",
        },
        "supervision": supervisor_semantics(),
        "search_excluded": True,
        "search_prohibition": integrated["search"],
    }
    return document


def recovery_semantics() -> dict:
    """The checkpoint-age story, stated the way Agent 3 measured it.

    Deliberately *not* "a crash loses at most fifteen minutes". Checkpoints are
    written at safe training boundaries, and no hot checkpoint is written during
    a collection, which ran 297-311 s in the rehearsal. Sealed games survive a
    later learner crash; what a crash can cost is un-checkpointed optimizer
    work.
    """
    from .phase14_contract import HOT_CHECKPOINT_SECONDS

    return {
        "hot_checkpoint_cadence_seconds": HOT_CHECKPOINT_SECONDS,
        "cadence_is_nominal": True,
        "writes_occur_at": "safe training boundaries, not on a timer interrupt",
        "no_write_during": "a collection unit (297-311 s observed in the rehearsal)",
        "sealed_games_survive_a_later_learner_crash": True,
        "what_a_crash_can_cost": "un-checkpointed optimizer work, which is repeated",
        "max_checkpoint_age_observed_seconds": 895.4,
        "observed_samples_above_cadence": 0,
        "observed_sample_count": 1067,
        "read_checkpoint_age_as": (
            "up to one cadence plus a collection, not up to one cadence"
        ),
        "games_replayed_across_two_rehearsal_crashes": 0,
        "optimizer_work_repeated_in_rehearsal_seconds": [197.6, 232.4],
        "evidence": "phase13_rehearsal_v1 sections 4 and 6",
    }


def rng_semantics() -> dict:
    """Captured, not restored — and why that is correct rather than a gap."""
    return {
        "global_rng_state_captured": True,
        "global_rng_state_restored": False,
        "reason": (
            "logical Phase 14 randomness uses explicit deterministic streams rather "
            "than ambient global RNG"
        ),
        "evidence": (
            "Phase 13 Agent 3 collected and trained the same iteration twice under two "
            "deliberately different global torch seeds: identical games, decisions, "
            "terminal results, bucket counts, updates, examples consumed, epoch plan "
            "and final model state digest "
            "f27ea740c4f13c7e5c8576adb1a789c84061cc6777ba7661d90f311cfa9cdf60"
        ),
        "redesigned_by_agent_4": False,
    }


def build_launch_manifest(
    *,
    storage=None,
    topology: "OperationalTopology | None" = None,
    scripts: "dict | None" = None,
) -> dict:
    """`phase14_launch_manifest_v1`: what this launch is allowed to be.

    The manifest is the thing a launch is checked against, so everything in it
    is either recomputed here from live code or read from an accepted artifact
    on disk. Nothing is transcribed from a report.
    """
    from .phase14_config import integrated_config_digest
    from .phase14_contract import (
        ANCHOR_CHECKPOINTS,
        ANCHOR_SHA256,
        FROZEN_CONTRACT_RELATIVE_PATH,
        SELECTION_PACK_DIGEST,
        SELECTION_RULE,
        SETUP_SELECTOR_CONFIG_SHA256,
        SETUP_SOURCE_IDENTITY,
        STARTING_CHECKPOINT,
        STARTING_CHECKPOINT_SHA256,
        STARTING_MODEL_STATE_DIGEST,
        contract_digest,
    )
    from .phase14_storage import Phase14Storage

    storage = storage or Phase14Storage.production()
    topology = topology or OperationalTopology.frozen()
    root = repository_root()
    scripts = dict(scripts or DEFAULT_SCRIPTS)
    config = final_training_config_document()
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    manifest = {
        "artifact": LAUNCH_MANIFEST_ARTIFACT,
        "launch_version": PHASE14_LAUNCH_VERSION,
        "phase": 13,
        "agent": 4,
        "status": "FROZEN",
        "built_utc": _utc_text(),
        "code": code_binding(),
        "upstream": {
            "agent1_contract": {
                "path": FROZEN_CONTRACT_RELATIVE_PATH,
                "sha256": file_sha256(root / FROZEN_CONTRACT_RELATIVE_PATH),
            },
            "agent2_integrated_config": {
                "path": "reports/phase13/phase13_integrated_training_config_v1.json",
                "sha256": file_sha256(
                    root / "reports/phase13/phase13_integrated_training_config_v1.json"
                ),
                "integrated_config_digest": integrated_config_digest(),
            },
            "agent3_rehearsal": {
                "path": "reports/phase13/phase13_rehearsal_v1.json",
                "sha256": file_sha256(root / "reports/phase13/phase13_rehearsal_v1.json"),
            },
            "starting_checkpoint": {
                "path": STARTING_CHECKPOINT,
                "sha256": STARTING_CHECKPOINT_SHA256,
                "observed_sha256": file_sha256(root / STARTING_CHECKPOINT),
                "model_state_digest": STARTING_MODEL_STATE_DIGEST,
            },
            "pool_anchors": {
                name: {
                    "path": ANCHOR_CHECKPOINTS[name],
                    "sha256": ANCHOR_SHA256[name],
                    "observed_sha256": file_sha256(root / ANCHOR_CHECKPOINTS[name]),
                }
                for name in sorted(ANCHOR_CHECKPOINTS)
            },
            "setup_source": {
                "identity": SETUP_SOURCE_IDENTITY,
                "selector_config_sha256": SETUP_SELECTOR_CONFIG_SHA256,
                "path": "reports/phase13/phase14_setup_source_v1.json",
                "file_sha256": file_sha256(
                    root / "reports/phase13/phase14_setup_source_v1.json"
                ),
            },
            "candidate_pack": {
                "path": "reports/phase13/phase14_checkpoint_selection_pack_v1.json",
                "pack_content_digest": SELECTION_PACK_DIGEST,
                "file_sha256": file_sha256(
                    root / "reports/phase13/phase14_checkpoint_selection_pack_v1.json"
                ),
            },
            "selection_rule": {
                "identity": SELECTION_RULE,
                "path": "reports/phase13/phase14_checkpoint_selection_rule_v1.json",
                "file_sha256": file_sha256(
                    root / "reports/phase13/phase14_checkpoint_selection_rule_v1.json"
                ),
            },
        },
        "phase14_contract_digest": contract_digest(),
        "integrated_config_digest": integrated_config_digest(),
        "phase14_final_training_config": {
            "artifact": FINAL_CONFIG_ARTIFACT,
            "path": FINAL_CONFIG_RELATIVE_PATH,
        },
        "phase14_final_training_config_digest": config_digest,
        "operational_topology": topology.to_dict(),
        "storage": {
            "external_volume": storage.external_root.parent.as_posix()
            if storage.external_root.parent != storage.external_root
            else str(storage.external_root),
            "external_run_directory": str(storage.external_root),
            "rollout_root": str(storage.rollout_root),
            "archive_root": str(storage.archive_root),
            "evaluation_root": str(storage.evaluation_root),
            "log_root": str(storage.log_root),
            "hot_checkpoint_root": str(storage.hot_root),
            "run_state_path": str(storage.run_state_path),
        },
        "scripts": {
            name: {"path": relative, "sha256": file_sha256(root / relative)}
            for name, relative in sorted(scripts.items())
        },
        "control_files": {
            "emergency_stop": str(emergency_stop_path(storage.external_root)),
            "integrity_failure": str(integrity_failure_path(storage.external_root)),
        },
        "deadline": {
            "duration_hours": 168,
            "derivation": "run_deadline_utc = actual launch UTC + 168 hours",
            "materialized": "exactly once, by the launch, and persisted",
            "run_start_utc": None,
            "run_deadline_utc": None,
            "note": (
                "these two fields are null by design: the absolute deadline cannot be "
                "known until Phase 14 launches, and the launch materializes them once"
            ),
        },
        "search_excluded": True,
    }
    # The build timestamp is excluded from the identity on purpose: rebuilding
    # the package over unchanged code should produce the *same* manifest
    # identity, so "the digest moved" means the bound content moved and nothing
    # else. `built_utc` stays in the document as provenance.
    identity = {key: value for key, value in manifest.items() if key != "built_utc"}
    manifest["launch_manifest_digest"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["launch_manifest_digest_excludes"] = ["built_utc"]
    return manifest


def write_launch_package(*, config_path=None, manifest_path=None, storage=None) -> dict:
    """Write both frozen artifacts and return their identities."""
    root = repository_root()
    config_target = Path(config_path or root / FINAL_CONFIG_RELATIVE_PATH)
    manifest_target = Path(manifest_path or root / LAUNCH_MANIFEST_RELATIVE_PATH)
    config = final_training_config_document()
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config["phase14_final_training_config_digest"] = config_digest
    config_target.parent.mkdir(parents=True, exist_ok=True)
    config_target.write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    manifest = build_launch_manifest(storage=storage)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return {
        "config_path": str(config_target),
        "config_digest": config_digest,
        "manifest_path": str(manifest_target),
        "manifest_digest": manifest["launch_manifest_digest"],
        "code_digest": manifest["code"]["code_digest"],
        "revision": manifest["code"]["git"]["revision"],
    }


# ---------------------------------------------------------------------------
# The host has to stay awake for 168 hours
# ---------------------------------------------------------------------------


def host_power_state() -> dict:
    """Whether this Mac will stay awake for a week, read from `pmset`.

    Not a fussy detail. The deadline is wall-clock: a machine that idle-sleeps
    at hour 3 does not pause the run, it *loses* the hours. `pmset` reports the
    configured idle-sleep minutes and, in parentheses, any power assertion
    currently preventing sleep — which is what `caffeinate` creates, and which
    is how the runbook's launch command satisfies this without a password.
    """
    try:
        finished = subprocess.run(
            ["pmset", "-g"], capture_output=True, text=True, check=False
        )
    except OSError as error:
        return {"known": False, "error": f"{type(error).__name__}: {error}"}
    if finished.returncode != 0:
        return {"known": False, "error": finished.stderr.strip()[:500]}
    settings = {}
    prevented_by = ""
    for line in finished.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        if key not in ("sleep", "disksleep", "displaysleep", "standby"):
            continue
        try:
            settings[key] = int(parts[1])
        except ValueError:
            continue
        if key == "sleep" and "prevented by" in line:
            prevented_by = line.split("prevented by", 1)[1].strip(" )")
    idle_sleep = settings.get("sleep")
    disk_sleep = settings.get("disksleep")
    return {
        "known": bool(settings),
        "settings": settings,
        "idle_sleep_minutes": idle_sleep,
        "disk_sleep_minutes": disk_sleep,
        "sleep_prevented_by": prevented_by,
        "will_stay_awake": bool(idle_sleep == 0 or prevented_by),
        "disk_will_stay_spun_up": bool(disk_sleep == 0 or prevented_by),
    }


def assert_host_stays_awake() -> dict:
    """Refuse to start a 168-hour run on a machine that will sleep."""
    state = host_power_state()
    if not state.get("known"):
        # Unknown is not the same as unsafe, and refusing on an unreadable
        # `pmset` would block a launch for the wrong reason. It is recorded.
        return {**state, "verified": False, "note": "pmset could not be read"}
    if not state["will_stay_awake"]:
        raise Phase14LaunchError(
            f"this Mac idle-sleeps after {state['idle_sleep_minutes']} minute(s) and no "
            "power assertion is preventing it. The 168-hour deadline is wall-clock: a "
            "machine that sleeps at hour 3 loses the hours, it does not pause them. "
            "Launch under `caffeinate -dimsu` (the runbook's launch command), or set "
            "`sudo pmset -a sleep 0 disksleep 0`"
        )
    return {**state, "verified": True}


def launch_semantics() -> dict:
    return {
        "launch_version": PHASE14_LAUNCH_VERSION,
        "code_binding": "git revision + tracked-tree state + content digest over the import closure",
        "repair_binding": "positive assertions that the accepted worker-pool repair is installed",
        "topology": "frozen from the Agent 3 rehearsal; outside the logical config digest",
        "emergency_stop": EMERGENCY_STOP_FILENAME,
        "integrity_failure": INTEGRITY_FAILURE_FILENAME,
        "namespace": PHASE14_NAMESPACE,
    }
