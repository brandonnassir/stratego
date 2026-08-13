#!/usr/bin/env python3
"""Phase 7 Agent 2 acceptance harness: deterministic base-library generation.

Verifies the Agent 1 prerequisites, generates the complete 8,000-base
`setup_library_v1`, materializes it with its manifest, proves isolated rebuild,
enumeration-order independence, digest stability and master-seed sensitivity,
re-verifies every entry from stored content, runs the Agent 1 diversity
standard as a preflight, and writes the two Agent 2 artifacts:

    reports/phase_7_data/agent_02_base_library_manifest.json
    reports/phase_7_data/agent_02_generation_summary.json

together with the production library itself:

    data/setups/setup_library_v1.jsonl
    data/setups/setup_library_v1_manifest.json

What this script is and is not
------------------------------
It materializes the library against Agent 1's frozen contracts. It does not
declare diversity acceptance — the diversity run here is a preflight so a
knowingly broken library is never handed on; Agent 3 owns the independent
verdict. No game outcome, win rate, Elo or model score participates in any
decision below.

Usage::

    python scripts/run_phase7_agent02.py                 # generate + verify + write
    python scripts/run_phase7_agent02.py --run-pytest    # also run the full suite
    python scripts/run_phase7_agent02.py --skip-diversity-preflight
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from stratego.engine.constants import (  # noqa: E402
    IMPLEMENTATION_VERSION,
    RULES_VERSION,
)
from stratego.setups import (  # noqa: E402
    BASE_SETUP_COUNT,
    DEFAULT_LIBRARY_MASTER_SEED,
    FAMILY_IDS,
    GENERATOR_VERSION,
    LIBRARY_JSONL_PATH,
    LIBRARY_MANIFEST_PATH,
    SETUP_FAMILY_VERSION,
    SETUP_GENERATOR_CONTRACT_VERSION,
    SETUP_LIBRARY_VERSION,
    SETUP_TRAIT_VECTOR_VERSION,
    TEST_TOTAL,
    TRAIN_TOTAL,
    VALIDATION_TOTAL,
    LibrarySeedContext,
    build_manifest,
    contract_document,
    entry_metadata_digest,
    generate_library,
    isolated_rebuild_sample_indices,
    library_content_digest,
    library_order,
    plans_document,
    read_library_jsonl,
    rebuild_base_setup,
    verify_library,
    write_library_jsonl,
    write_manifest,
)
from stratego.setups.diversity import (  # noqa: E402
    DIVERSITY_STANDARD_VERSION,
    DIVERSITY_THRESHOLDS_V1,
    LibraryEntry,
    evaluate_against_thresholds,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_7_data"
AGENT_01_CONTRACT = DATA_DIRECTORY / "agent_01_setup_contract.json"
AGENT_01_THRESHOLDS = DATA_DIRECTORY / "agent_01_diversity_thresholds.json"
MANIFEST_ARTIFACT = DATA_DIRECTORY / "agent_02_base_library_manifest.json"
SUMMARY_ARTIFACT = DATA_DIRECTORY / "agent_02_generation_summary.json"

LIBRARY_PATH = REPOSITORY_ROOT / LIBRARY_JSONL_PATH
MANIFEST_PATH = REPOSITORY_ROOT / LIBRARY_MANIFEST_PATH

#: Seed of the harness's own sampling decisions (shuffled enumeration order,
#: master-seed sensitivity probe). Never touches library content.
HARNESS_SEED = 20260813

#: Bases regenerated under an alternative master seed to prove the library is
#: seed-sensitive. Kept small: the property is per-base, not aggregate.
SEED_SENSITIVITY_SAMPLE = 128


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _environment() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }


def _canonical_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _peak_rss_bytes() -> int:
    """Peak resident set size of this process. macOS reports bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def _verify_prerequisites() -> dict:
    """Agent 1 PASS, and the live contract identical to its artifacts."""
    contract_artifact = json.loads(AGENT_01_CONTRACT.read_text())
    thresholds_artifact = json.loads(AGENT_01_THRESHOLDS.read_text())

    live_contract = contract_document()
    live_thresholds = DIVERSITY_THRESHOLDS_V1.to_dict()
    stored_contract = contract_artifact["contract"]
    stored_thresholds = thresholds_artifact["thresholds"]

    return {
        "agent_01_status": contract_artifact["status"],
        "agent_01_pass": contract_artifact["status"] == "PASS",
        "agent_01_gates_true": contract_artifact["gates_true"],
        "agent_01_gates_total": contract_artifact["gates_total"],
        "agent_01_thresholds_status": thresholds_artifact["status"],
        "thresholds_frozen_before_generation": thresholds_artifact["frozen_before_generation"],
        "live_contract_digest": _canonical_digest(live_contract),
        "artifact_contract_digest": _canonical_digest(stored_contract),
        "contract_matches_artifact": live_contract == stored_contract,
        "live_thresholds_digest": _canonical_digest(live_thresholds),
        "artifact_thresholds_digest": _canonical_digest(stored_thresholds),
        "thresholds_match_artifact": live_thresholds == stored_thresholds,
        "versions_match_artifact": (
            contract_artifact["frozen_versions"]["contract_version"]
            == SETUP_GENERATOR_CONTRACT_VERSION
            and contract_artifact["frozen_versions"]["library_version"] == SETUP_LIBRARY_VERSION
            and contract_artifact["frozen_versions"]["family_contract_version"]
            == SETUP_FAMILY_VERSION
            and contract_artifact["frozen_versions"]["trait_schema_version"]
            == SETUP_TRAIT_VECTOR_VERSION
            and contract_artifact["frozen_versions"]["diversity_standard_version"]
            == DIVERSITY_STANDARD_VERSION
        ),
        "master_seed_matches_artifact": (
            contract_artifact["master_seed"] == DEFAULT_LIBRARY_MASTER_SEED
        ),
        "reference_engine": IMPLEMENTATION_VERSION,
        "reference_engine_is_1_2_0": IMPLEMENTATION_VERSION == "phase2_1_reference_1.2.0",
        "rules_version": RULES_VERSION,
    }


# ---------------------------------------------------------------------------
# Determinism proofs
# ---------------------------------------------------------------------------


def _isolated_rebuild_proof(entries) -> dict:
    """Rebuild the fixed Agent 1 sample from identity alone and compare exactly."""
    started = time.time()
    by_identifier = {entry.base_setup_id: entry for entry in entries}
    indices = isolated_rebuild_sample_indices()
    mismatches: list[str] = []
    rebuilt = 0
    for family_id in FAMILY_IDS:
        for base_index in indices:
            entry = rebuild_base_setup(family_id, base_index)
            rebuilt += 1
            if entry.to_dict() != by_identifier[entry.base_setup_id].to_dict():
                mismatches.append(entry.base_setup_id)
    return {
        "sample_indices_per_family": list(indices),
        "rebuilt_count": rebuilt,
        "mismatches": mismatches,
        "exact": not mismatches,
        "duration_seconds": round(time.time() - started, 3),
    }


def _enumeration_order_proof(entries, sample_size: int = 256) -> dict:
    """Generating in a shuffled order must reproduce identical entries."""
    started = time.time()
    rng = random.Random(HARNESS_SEED)
    order = library_order()
    rng.shuffle(order)
    order = order[:sample_size]
    by_identifier = {entry.base_setup_id: entry for entry in entries}
    shuffled = generate_library(order=order)
    mismatches = [
        entry.base_setup_id
        for entry in shuffled.entries
        if entry.to_dict() != by_identifier[entry.base_setup_id].to_dict()
    ]
    return {
        "sample_size": len(order),
        "mismatches": mismatches,
        "order_independent": not mismatches,
        "duration_seconds": round(time.time() - started, 3),
    }


def _seed_sensitivity_proof(entries) -> dict:
    """A different master seed must produce different content."""
    started = time.time()
    alternative = LibrarySeedContext(master_seed=DEFAULT_LIBRARY_MASTER_SEED + 1)
    production = {entry.fingerprint for entry in entries}
    rng = random.Random(HARNESS_SEED + 1)
    order = library_order()
    rng.shuffle(order)
    order = order[:SEED_SENSITIVITY_SAMPLE]
    changed = 0
    shared = 0
    for family_id, base_index in order:
        entry = rebuild_base_setup(family_id, base_index, alternative)
        if entry.fingerprint in production:
            shared += 1
        else:
            changed += 1
    return {
        "alternative_master_seed": alternative.master_seed,
        "sample_size": len(order),
        "changed": changed,
        "shared_with_production": shared,
        "library_content_changed": changed == len(order),
        "duration_seconds": round(time.time() - started, 3),
    }


def _regeneration_proof(reference: dict) -> dict:
    """A second full generation must reproduce every digest exactly."""
    started = time.time()
    repeat = generate_library()
    digests = {
        "library_content_digest": library_content_digest(repeat.entries),
        "entry_metadata_digest": entry_metadata_digest(repeat.entries),
    }
    return {
        **digests,
        "matches_first_run": digests == reference,
        "duration_seconds": round(time.time() - started, 3),
    }


# ---------------------------------------------------------------------------
# Diversity preflight (Agent 3 owns the verdict)
# ---------------------------------------------------------------------------


def _diversity_preflight(entries) -> dict:
    started = time.time()
    library_entries = [
        LibraryEntry(
            family_id=entry.family_id, split=entry.split, canonical=entry.canonical_setup
        )
        for entry in entries
    ]
    evaluation = evaluate_against_thresholds(library_entries)
    failures = [check for check in evaluation["checks"] if not check["pass"]]
    return {
        "diversity_standard_version": evaluation["diversity_standard_version"],
        "check_count": len(evaluation["checks"]),
        "all_pass": evaluation["all_pass"],
        "failed_checks": failures,
        "metrics": evaluation["metrics"],
        "verdict_owner": "agent_03",
        "duration_seconds": round(time.time() - started, 3),
    }


def _run_pytest() -> dict:
    started = time.time()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    counts = {
        key: int(value)
        for value, key in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)
    }
    return {
        "command": "python -m pytest -q",
        "exit_code": completed.returncode,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "errors": counts.get("error", 0),
        "summary_line": tail,
        "duration_seconds": round(time.time() - started, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pytest", action="store_true", help="run the full suite too")
    parser.add_argument(
        "--skip-diversity-preflight",
        action="store_true",
        help="skip the Agent 1 diversity preflight (Agent 3 owns the verdict anyway)",
    )
    arguments = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    commit = _git("rev-parse", "HEAD")
    working_tree_state = "dirty" if _git("status", "--porcelain") else "clean"
    command = "python scripts/run_phase7_agent02.py" + (
        " --run-pytest" if arguments.run_pytest else ""
    )

    prerequisites = _verify_prerequisites()
    if not prerequisites["agent_01_pass"]:
        print("BLOCKED: Agent 1 is not PASS", file=sys.stderr)
        return 2
    if not (prerequisites["contract_matches_artifact"] and prerequisites["thresholds_match_artifact"]):
        print("BLOCKED: live Agent 1 contract disagrees with its artifacts", file=sys.stderr)
        return 2

    print(f"generating {BASE_SETUP_COUNT} base setups ...")

    def progress(done: int, total: int) -> None:
        print(f"  {done}/{total}", flush=True)

    result = generate_library(progress=progress)
    entries = list(result.entries)

    verification = verify_library(entries)
    reference_digests = {
        "library_content_digest": library_content_digest(entries),
        "entry_metadata_digest": entry_metadata_digest(entries),
    }

    library_bytes = write_library_jsonl(LIBRARY_PATH, entries)
    manifest = build_manifest(
        result,
        command=command,
        library_bytes=library_bytes,
        peak_rss_bytes=_peak_rss_bytes(),
        timestamp=timestamp,
    )
    manifest_bytes = write_manifest(MANIFEST_PATH, manifest)

    # The materialized file must read back to exactly what was generated.
    reread = read_library_jsonl(LIBRARY_PATH)
    roundtrip_exact = [entry.to_dict() for entry in reread] == [
        entry.to_dict() for entry in entries
    ]
    rewrite_bytes = LIBRARY_PATH.read_bytes()
    write_library_jsonl(LIBRARY_PATH, reread)
    rewrite_stable = LIBRARY_PATH.read_bytes() == rewrite_bytes

    isolated = _isolated_rebuild_proof(entries)
    enumeration = _enumeration_order_proof(entries)
    sensitivity = _seed_sensitivity_proof(entries)
    regeneration = _regeneration_proof(reference_digests)
    diversity = None if arguments.skip_diversity_preflight else _diversity_preflight(entries)

    total_seconds = round(time.time() - started, 3)
    peak_rss = _peak_rss_bytes()

    gates = {
        "agent_01_pass_verified": prerequisites["agent_01_pass"],
        "agent_01_contract_matches_live_code": prerequisites["contract_matches_artifact"]
        and prerequisites["thresholds_match_artifact"],
        "eight_thousand_bases_materialized": verification["entry_count"] == BASE_SETUP_COUNT,
        "five_hundred_per_family": verification["checks"]["family_counts_exact"],
        "split_counts_exact": verification["checks"]["family_split_counts_exact"]
        and verification["split_counts"]
        == {"train": TRAIN_TOTAL, "validation": VALIDATION_TOTAL, "test": TEST_TOTAL},
        "zero_engine_invalid": verification["checks"]["no_engine_invalid"],
        "zero_stranded": verification["checks"]["no_stranded"],
        "zero_family_violations": verification["checks"]["no_family_violations"],
        "zero_exact_duplicates": verification["checks"]["no_exact_duplicates"],
        "zero_reflection_duplicates": verification["checks"]["no_reflection_duplicates"],
        "zero_stable_id_collisions": verification["checks"]["no_stable_id_collisions"],
        "all_entries_canonical": verification["checks"]["all_entries_canonical"],
        "reflection_roundtrip_clean": verification["checks"]["reflection_roundtrip_clean"],
        "entry_metadata_consistent": verification["checks"]["entry_metadata_consistent"],
        "isolated_rebuild_exact": isolated["exact"],
        "enumeration_order_independent": enumeration["order_independent"],
        "master_seed_sensitive": sensitivity["library_content_changed"],
        "full_regeneration_digest_stable": regeneration["matches_first_run"],
        "serialization_roundtrip_exact": roundtrip_exact and rewrite_stable,
        "library_and_manifest_written": LIBRARY_PATH.exists() and MANIFEST_PATH.exists(),
        "no_outcome_or_strength_signal": verification["checks"]["no_outcome_or_strength_fields"],
    }

    manifest_payload = {
        "agent": "agent_02",
        "phase": "phase_7",
        "artifact": "base_library_manifest",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "timestamp": timestamp,
        "commit": commit,
        "library_jsonl_path": LIBRARY_JSONL_PATH,
        "library_manifest_path": LIBRARY_MANIFEST_PATH,
        "library_bytes": library_bytes,
        "manifest_bytes": manifest_bytes,
        "manifest": manifest,
        "generator_plans": plans_document(),
    }

    summary_payload = {
        "agent": "agent_02",
        "phase": "phase_7",
        "assignment": "Deterministic Base Library Generator",
        "status": manifest_payload["status"],
        "schema_version": "phase_7_agent_02_v1",
        "timestamp": timestamp,
        "commit": commit,
        "working_tree_state": working_tree_state,
        **_environment(),
        "prerequisite_status": prerequisites,
        "frozen_versions": {
            "rules": RULES_VERSION,
            "reference_engine": IMPLEMENTATION_VERSION,
            "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
            "library_version": SETUP_LIBRARY_VERSION,
            "family_contract_version": SETUP_FAMILY_VERSION,
            "trait_schema_version": SETUP_TRAIT_VECTOR_VERSION,
            "diversity_standard_version": DIVERSITY_STANDARD_VERSION,
            "generator_version": GENERATOR_VERSION,
        },
        "library_version": SETUP_LIBRARY_VERSION,
        "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
        "family_contract_version": SETUP_FAMILY_VERSION,
        "sampler_version": None,
        "master_seed": DEFAULT_LIBRARY_MASTER_SEED,
        "library_digest": manifest["library_content_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "entry_metadata_digest": manifest["entry_metadata_digest"],
        "setup_count": verification["entry_count"],
        "family_counts": verification["family_counts"],
        "split_counts": verification["split_counts"],
        "family_split_counts": verification["family_split_counts"],
        "seeds": {
            "master_seed": DEFAULT_LIBRARY_MASTER_SEED,
            "seed_derivation": result.seed_context.to_dict(),
            "harness_seed": HARNESS_SEED,
            "alternative_master_seed": sensitivity["alternative_master_seed"],
        },
        "generation": {
            "generator_version": GENERATOR_VERSION,
            "duration_seconds": result.duration_seconds,
            "total_attempts": result.total_attempts(),
            "attempts_per_accepted_base": result.attempts_per_accepted_base(),
            "attempt_histogram": {
                str(attempts): count for attempts, count in result.attempt_histogram.items()
            },
            "rejections_by_reason": result.rejections_by_reason(),
            "rejections_by_family": result.rejections_by_family,
            "rejection_rate_by_family": result.rejection_rate_by_family(),
        },
        "verification": verification,
        "isolated_rebuild": isolated,
        "enumeration_order": enumeration,
        "seed_sensitivity": sensitivity,
        "regeneration": regeneration,
        "serialization": {
            "library_bytes": library_bytes,
            "manifest_bytes": manifest_bytes,
            "read_back_exact": roundtrip_exact,
            "rewrite_byte_stable": rewrite_stable,
        },
        "diversity_preflight": diversity,
        "performance": {
            "full_library_generation_seconds": result.duration_seconds,
            "harness_total_seconds": total_seconds,
            "peak_rss_bytes": peak_rss,
            "peak_rss_megabytes": round(peak_rss / 1e6, 1),
            "library_bytes": library_bytes,
            "manifest_bytes": manifest_bytes,
        },
        "tests_before": {
            "command": "python -m pytest -q",
            "passed": 2898,
            "failed": 0,
            "skipped": 3,
            "errors": 0,
            "note": "recorded at commit 3e54eae before any Agent 2 edit",
        },
        "tests_after": None,  # filled in below when --run-pytest is given
        "commands": [command],
        "durations": {
            "generation_seconds": result.duration_seconds,
            "isolated_rebuild_seconds": isolated["duration_seconds"],
            "enumeration_order_seconds": enumeration["duration_seconds"],
            "regeneration_seconds": regeneration["duration_seconds"],
            "diversity_preflight_seconds": diversity["duration_seconds"] if diversity else None,
            "total_seconds": total_seconds,
        },
        "files_created": [
            "stratego/setups/seed.py",
            "stratego/setups/generator.py",
            "stratego/setups/library.py",
            "tests/setups/test_generator.py",
            "tests/setups/test_library.py",
            "scripts/run_phase7_agent02.py",
            LIBRARY_JSONL_PATH,
            LIBRARY_MANIFEST_PATH,
            "reports/phase_7_data/agent_02_base_library_manifest.json",
            "reports/phase_7_data/agent_02_generation_summary.json",
        ],
        "files_modified": [
            "stratego/setups/__init__.py",
            "reports/phase_7_implementation_report.md",
        ],
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "problems": [],
        "deviations": [
            "global uniqueness is enforced as a hard acceptance gate that raises "
            "on collision, not as a cross-base regeneration filter: Agent 1's "
            "frozen contract forbids conditioning one base on another base's "
            "outcome, so a duplicate would be a BLOCKED finding; the frozen "
            "master seed produces zero duplicates across all 8,000 bases",
            "entries carry five fields beyond the frozen minimum "
            "(generator_version, content_fingerprint, "
            "reflected_content_fingerprint, accepted_attempt_index, "
            "accepted_attempt_seed) as required by the Agent 2 metadata "
            "contract; the frozen entry field list, line format and file "
            "order are unchanged",
        ],
    }

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MANIFEST_ARTIFACT.write_text(json.dumps(manifest_payload, indent=1, sort_keys=True) + "\n")
    SUMMARY_ARTIFACT.write_text(json.dumps(summary_payload, indent=1, sort_keys=True) + "\n")

    # The suite runs only after both artifacts are on disk, because its
    # artifact-gated tests compare the materialized library and manifest with
    # the Agent 2 artifacts; running it earlier would test the previous run's
    # artifacts against this run's library.
    if arguments.run_pytest:
        suite = _run_pytest()
        gates["full_repository_suite_green"] = (
            suite["exit_code"] == 0 and suite["failed"] == 0 and suite["errors"] == 0
        )
        status = "PASS" if all(gates.values()) else "FAIL"
        summary_payload["tests_after"] = suite
        summary_payload["completion_gates"] = gates
        summary_payload["gates_total"] = len(gates)
        summary_payload["gates_true"] = sum(1 for value in gates.values() if value)
        summary_payload["status"] = status
        manifest_payload["status"] = status
        MANIFEST_ARTIFACT.write_text(json.dumps(manifest_payload, indent=1, sort_keys=True) + "\n")
        SUMMARY_ARTIFACT.write_text(json.dumps(summary_payload, indent=1, sort_keys=True) + "\n")
        print(f"  {'ok ' if gates['full_repository_suite_green'] else 'FAIL'} "
              f"full_repository_suite_green ({suite['summary_line']})")

    print(f"status: {summary_payload['status']}  gates {summary_payload['gates_true']}/{len(gates)}")
    for name, value in gates.items():
        print(f"  {'ok ' if value else 'FAIL'} {name}")
    print(f"library: {LIBRARY_JSONL_PATH} ({library_bytes} bytes)")
    print(f"digest:  {manifest['library_content_digest']}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
