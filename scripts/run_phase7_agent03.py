#!/usr/bin/env python3
"""Phase 7 Agent 3 acceptance harness: exhaustive independent library audit.

Verifies the Agent 1/Agent 2 prerequisites and the production-library handoff
digests, then audits all 8,000 bases and all 8,000 reflections exhaustively —
engine legality/inventory/placement, initial mobility, identity and
canonicalization, split correctness, family contracts, global duplicates,
cross-split near-duplicate leakage, Agent 1's frozen diversity thresholds,
serialization/reflection round trips, and the descriptive family
overlap/confusion matrix — recomputing every fact from the materialized JSONL
plus the frozen engine and contracts, never from Agent 2's counters. Writes:

    reports/phase_7_data/agent_03_library_audit.json
    reports/phase_7_data/agent_03_family_metrics.csv
    reports/phase_7_data/agent_03_similarity_audit.csv

What this script is and is not
------------------------------
It is an audit. It does not modify the production library, does not weaken or
reinterpret any Agent 1 threshold, and does not repair anything: a library
that fails a frozen threshold is reported FAIL, and a broken prerequisite or
handoff digest is reported BLOCKED. No game outcome, win rate, Elo or model
score participates in any decision below.

Usage::

    python scripts/run_phase7_agent03.py                 # audit + write artifacts
    python scripts/run_phase7_agent03.py --run-pytest    # also run the full suite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
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
    contract_document,
    read_library_jsonl,
    read_manifest,
)
from stratego.setups.audit import (  # noqa: E402
    AUDIT_VERSION,
    CROSS_SPLIT_FLOOR,
    NEAR_DUPLICATE_DISTANCE,
    audit_library,
)
from stratego.setups.diversity import (  # noqa: E402
    DIVERSITY_STANDARD_VERSION,
    DIVERSITY_THRESHOLDS_V1,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_7_data"
AGENT_01_CONTRACT = DATA_DIRECTORY / "agent_01_setup_contract.json"
AGENT_01_THRESHOLDS = DATA_DIRECTORY / "agent_01_diversity_thresholds.json"
AGENT_02_SUMMARY = DATA_DIRECTORY / "agent_02_generation_summary.json"
AGENT_02_MANIFEST = DATA_DIRECTORY / "agent_02_base_library_manifest.json"

AUDIT_ARTIFACT = DATA_DIRECTORY / "agent_03_library_audit.json"
FAMILY_METRICS_CSV = DATA_DIRECTORY / "agent_03_family_metrics.csv"
SIMILARITY_CSV = DATA_DIRECTORY / "agent_03_similarity_audit.csv"

LIBRARY_PATH = REPOSITORY_ROOT / LIBRARY_JSONL_PATH
MANIFEST_PATH = REPOSITORY_ROOT / LIBRARY_MANIFEST_PATH


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
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


# ---------------------------------------------------------------------------
# Prerequisites: Agents 1 and 2 PASS, live contracts frozen, handoff digests
# ---------------------------------------------------------------------------


def _verify_prerequisites() -> dict:
    contract_artifact = json.loads(AGENT_01_CONTRACT.read_text())
    thresholds_artifact = json.loads(AGENT_01_THRESHOLDS.read_text())
    summary_artifact = json.loads(AGENT_02_SUMMARY.read_text())
    manifest_artifact = json.loads(AGENT_02_MANIFEST.read_text())

    live_contract = contract_document()
    live_thresholds = DIVERSITY_THRESHOLDS_V1.to_dict()

    return {
        "agent_01_status": contract_artifact["status"],
        "agent_01_pass": contract_artifact["status"] == "PASS",
        "agent_01_thresholds_status": thresholds_artifact["status"],
        "agent_01_thresholds_pass": thresholds_artifact["status"] == "PASS",
        "thresholds_frozen_before_generation": thresholds_artifact[
            "frozen_before_generation"
        ],
        "agent_02_status": summary_artifact["status"],
        "agent_02_pass": summary_artifact["status"] == "PASS",
        "agent_02_gates": f"{summary_artifact['gates_true']}/{summary_artifact['gates_total']}",
        "live_contract_digest": _canonical_digest(live_contract),
        "artifact_contract_digest": _canonical_digest(contract_artifact["contract"]),
        "contract_matches_artifact": live_contract == contract_artifact["contract"],
        "live_thresholds_digest": _canonical_digest(live_thresholds),
        "artifact_thresholds_digest": _canonical_digest(
            thresholds_artifact["thresholds"]
        ),
        "thresholds_match_artifact": live_thresholds
        == thresholds_artifact["thresholds"],
        "handoff_digests": {
            "library_content_digest": summary_artifact["library_digest"],
            "entry_metadata_digest": summary_artifact["entry_metadata_digest"],
            "manifest_digest": summary_artifact["manifest_digest"],
        },
        "agent_02_artifact_manifest_digest": manifest_artifact["manifest"][
            "manifest_digest"
        ],
        "master_seed_matches": contract_artifact["master_seed"]
        == DEFAULT_LIBRARY_MASTER_SEED,
        "reference_engine": IMPLEMENTATION_VERSION,
        "reference_engine_is_1_2_0": IMPLEMENTATION_VERSION
        == "phase2_1_reference_1.2.0",
        "rules_version": RULES_VERSION,
        "library_file_exists": LIBRARY_PATH.exists(),
        "manifest_file_exists": MANIFEST_PATH.exists(),
    }


# ---------------------------------------------------------------------------
# CSV artifacts
# ---------------------------------------------------------------------------


def _write_family_metrics_csv(result: dict) -> int:
    """Per-family metrics: every frozen threshold plus descriptive context.

    Threshold rows show `measured / required / pass`; report-only rows carry
    `required = report-only` and an empty pass cell, so the CSV distinguishes
    the frozen standard from descriptive context without inventing new gates.
    """
    thresholds = DIVERSITY_THRESHOLDS_V1
    metrics = result["thresholds"]["metrics"]
    similarity = result["similarity"]
    overlap = result["overlap"]

    check_lookup = {
        check["check"]: check for check in result["thresholds"]["checks"]
    }

    rows: list[dict] = []

    def add(family_id, metric, measured, required, kind, passed) -> None:
        rows.append(
            {
                "family_id": family_id,
                "metric": metric,
                "measured": measured,
                "required": required,
                "threshold_kind": kind,
                "pass": "" if passed is None else str(bool(passed)).lower(),
            }
        )

    for family_id in FAMILY_IDS:
        within = similarity["within_family"][family_id]
        entropy = metrics["entropy"]["per_family"][family_id]
        traits = metrics["trait_diversity"]["per_family"][family_id]
        row = overlap["matrix"][family_id]

        named = f"{family_id}:min_within_family_nn_distance"
        add(family_id, "min_within_family_nn_distance",
            within["min_nn_distance"],
            f">= {thresholds.min_within_family_nn_distance}", "floor",
            check_lookup[named]["pass"])
        named = f"{family_id}:within_family_near_duplicate_fraction"
        add(family_id, "near_duplicate_pair_fraction",
            within["near_duplicate_pair_fraction"],
            f"<= {thresholds.max_within_family_near_duplicate_fraction}", "ceiling",
            check_lookup[named]["pass"])
        add(family_id, "near_duplicate_pairs_below_10",
            within["near_duplicate_pairs"], "report-only", "descriptive", None)
        add(family_id, "mean_per_square_entropy_bits",
            entropy["mean_per_square_entropy_bits"],
            f">= {thresholds.min_family_mean_per_square_entropy_bits}", "floor",
            check_lookup[f"{family_id}:mean_per_square_entropy_bits"]["pass"])
        add(family_id, "flag_folded_support", entropy["flag_folded_support"],
            f">= {thresholds.min_flag_folded_support[family_id]}", "floor",
            check_lookup[f"{family_id}:flag_folded_support"]["pass"])
        add(family_id, "bomb_folded_support", entropy["bomb_folded_support"],
            f">= {thresholds.min_bomb_folded_support}", "floor",
            check_lookup[f"{family_id}:bomb_folded_support"]["pass"])
        add(family_id, "scout_folded_support", entropy["scout_folded_support"],
            f">= {thresholds.min_scout_folded_support}", "floor",
            check_lookup[f"{family_id}:scout_folded_support"]["pass"])
        add(family_id, "miner_folded_support", entropy["miner_folded_support"],
            f">= {thresholds.min_miner_folded_support}", "floor",
            check_lookup[f"{family_id}:miner_folded_support"]["pass"])
        add(family_id, "high_rank_folded_support",
            entropy["high_rank_folded_support"],
            f">= {thresholds.min_high_rank_folded_support}", "floor",
            check_lookup[f"{family_id}:high_rank_folded_support"]["pass"])
        add(family_id, "distinct_trait_vectors", traits["distinct_trait_vectors"],
            f">= {thresholds.min_distinct_trait_vectors_per_family}", "floor",
            check_lookup[f"{family_id}:distinct_trait_vectors"]["pass"])
        add(family_id, "distinct_bomb_rank_histograms",
            traits["distinct_bomb_rank_histograms"],
            f">= {thresholds.min_distinct_bomb_rank_histograms_per_family}", "floor",
            check_lookup[f"{family_id}:distinct_bomb_rank_histograms"]["pass"])
        add(family_id, "distinct_scout_rank_histograms",
            traits["distinct_scout_rank_histograms"],
            f">= {thresholds.min_distinct_scout_rank_histograms_per_family}", "floor",
            check_lookup[f"{family_id}:distinct_scout_rank_histograms"]["pass"])
        add(family_id, "family_self_satisfaction", row[family_id],
            f"== {thresholds.required_self_satisfaction}", "exact",
            check_lookup[f"{family_id}:self_satisfaction"]["pass"])

        add(family_id, "within_family_nn_median", within["nn"]["median"],
            "report-only", "descriptive", None)
        add(family_id, "within_family_nn_max", within["nn"]["max"],
            "report-only", "descriptive", None)
        add(family_id, "within_family_pairwise_median",
            within["pairwise"]["median"], "report-only", "descriptive", None)
        largest = max(
            (
                (target, fraction)
                for target, fraction in row.items()
                if target != family_id
            ),
            key=lambda item: item[1],
        )
        add(family_id, "largest_secondary_overlap",
            f"{largest[0]}:{largest[1]}", "report-only", "descriptive", None)

    global_check = check_lookup["global_mean_per_square_entropy_bits"]
    add("ALL", "global_mean_per_square_entropy_bits", global_check["observed"],
        f">= {thresholds.min_global_mean_per_square_entropy_bits}", "floor",
        global_check["pass"])
    add("ALL", "global_min_pairwise_distance",
        similarity["global_min_pairwise_distance"],
        f">= {thresholds.min_global_pairwise_distance}", "floor",
        check_lookup["global_min_pairwise_distance"]["pass"])
    add("ALL", "cross_split_min_nn_distance",
        similarity["cross_split_min_nn_distance"],
        f">= {thresholds.min_cross_split_nn_distance}", "floor",
        check_lookup["cross_split_min_nn_distance"]["pass"])

    with FAMILY_METRICS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family_id", "metric", "measured", "required",
                        "threshold_kind", "pass"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_similarity_csv(result: dict) -> int:
    """Every similarity scope: global, within-family, and cross-split pairs."""
    similarity = result["similarity"]
    rows: list[dict] = []

    def add(scope, family_id, split_a, split_b, stats, nn_stats,
            below_floor, below_near, offending) -> None:
        rows.append(
            {
                "scope": scope,
                "family_id": family_id,
                "split_a": split_a,
                "split_b": split_b,
                "pair_count": stats["count"],
                "min": stats["min"],
                "p1": stats["p1"],
                "p5": stats["p5"],
                "p25": stats["p25"],
                "median": stats["median"],
                "mean": stats["mean"],
                "max": stats["max"],
                "nn_min": nn_stats["min"] if nn_stats else "",
                "nn_median": nn_stats["median"] if nn_stats else "",
                "nn_max": nn_stats["max"] if nn_stats else "",
                f"pairs_below_{CROSS_SPLIT_FLOOR}": below_floor,
                f"pairs_below_{NEAR_DUPLICATE_DISTANCE}": below_near,
                "offending_ids": ";".join(
                    f"{pair['a']}~{pair['b']}@{pair['class_distance']}"
                    for pair in offending
                ),
            }
        )

    global_stats = similarity["global"]
    add("global_pairwise", "ALL", "all", "all", global_stats["pairwise"],
        global_stats["nn"], "", "", [])

    for family_id, within in similarity["within_family"].items():
        add("within_family", family_id, "all", "all", within["pairwise"],
            within["nn"], "", within["near_duplicate_pairs"],
            within["offending_pairs"])

    for pair_name, scopes in similarity["cross_split"].items():
        split_a, split_b = pair_name.split("__")
        for scope_name, stats in scopes.items():
            add(
                "cross_split_global" if scope_name == "global" else "cross_split_family",
                "ALL" if scope_name == "global" else scope_name,
                split_a,
                split_b,
                stats["pairwise"],
                stats["nn_a_to_b"],
                stats["pairs_below_cross_split_floor"],
                stats["pairs_below_near_duplicate"],
                stats["offending_pairs"],
            )

    with SIMILARITY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Full suite
# ---------------------------------------------------------------------------


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
    arguments = parser.parse_args()

    started = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    commit = _git("rev-parse", "HEAD")
    working_tree_state = "dirty" if _git("status", "--porcelain") else "clean"
    command = "python scripts/run_phase7_agent03.py" + (
        " --run-pytest" if arguments.run_pytest else ""
    )

    prerequisites = _verify_prerequisites()
    blocked_reasons = []
    if not prerequisites["agent_01_pass"] or not prerequisites["agent_01_thresholds_pass"]:
        blocked_reasons.append("Agent 1 is not PASS")
    if not prerequisites["agent_02_pass"]:
        blocked_reasons.append("Agent 2 is not PASS")
    if not prerequisites["contract_matches_artifact"]:
        blocked_reasons.append("live contract disagrees with the Agent 1 artifact")
    if not prerequisites["thresholds_match_artifact"]:
        blocked_reasons.append("live thresholds disagree with the Agent 1 artifact")
    if not prerequisites["library_file_exists"] or not prerequisites["manifest_file_exists"]:
        blocked_reasons.append("production library files are missing")

    library_bytes_before = LIBRARY_PATH.read_bytes() if LIBRARY_PATH.exists() else b""
    manifest_bytes_before = MANIFEST_PATH.read_bytes() if MANIFEST_PATH.exists() else b""

    raw_text = library_bytes_before.decode("utf-8") if library_bytes_before else ""
    entries = read_library_jsonl(LIBRARY_PATH) if not blocked_reasons else []
    manifest = read_manifest(MANIFEST_PATH) if not blocked_reasons else {}

    # The handoff digest check happens before any audit stage: a mismatch is
    # BLOCKED (wrong input), not FAIL (bad library).
    if not blocked_reasons:
        from stratego.setups.library import (
            entry_metadata_digest,
            library_content_digest,
            manifest_digest,
        )

        handoff = prerequisites["handoff_digests"]
        digest_match = {
            "library_content_digest": library_content_digest(entries)
            == handoff["library_content_digest"],
            "entry_metadata_digest": entry_metadata_digest(entries)
            == handoff["entry_metadata_digest"],
            "manifest_digest": manifest_digest(manifest) == handoff["manifest_digest"],
        }
        prerequisites["handoff_digest_match"] = digest_match
        if not all(digest_match.values()):
            blocked_reasons.append(
                "production library digests do not match the Agent 2 handoff"
            )

    if blocked_reasons:
        payload = {
            "agent": "agent_03",
            "phase": "phase_7",
            "assignment": "Exhaustive Library Audit",
            "status": "BLOCKED",
            "schema_version": "phase_7_agent_03_v1",
            "timestamp": timestamp,
            "commit": commit,
            "working_tree_state": working_tree_state,
            **_environment(),
            "prerequisite_status": prerequisites,
            "problems": blocked_reasons,
        }
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        AUDIT_ARTIFACT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print("BLOCKED:", "; ".join(blocked_reasons), file=sys.stderr)
        return 2

    print(f"auditing {len(entries)} bases and {len(entries)} reflections ...")
    audit = audit_library(
        entries,
        manifest=manifest,
        raw_text=raw_text,
        expected_digests=prerequisites["handoff_digests"],
    )

    # The audit must not have touched the production files.
    library_untouched = LIBRARY_PATH.read_bytes() == library_bytes_before
    manifest_untouched = MANIFEST_PATH.read_bytes() == manifest_bytes_before

    total_seconds = round(time.time() - started, 3)
    peak_rss = _peak_rss_bytes()

    threshold_result = audit["thresholds"]
    similarity = audit["similarity"]
    per_base = audit["per_base"]

    gates = {
        "agent_01_pass_verified": prerequisites["agent_01_pass"]
        and prerequisites["agent_01_thresholds_pass"],
        "agent_02_pass_verified": prerequisites["agent_02_pass"],
        "contract_and_thresholds_frozen": prerequisites["contract_matches_artifact"]
        and prerequisites["thresholds_match_artifact"],
        "handoff_digests_verified_before_audit": all(
            prerequisites["handoff_digest_match"].values()
        ),
        "counts_exact": audit["counts"]["all_exact"],
        "all_bases_engine_valid": not per_base["engine_failures"]
        and not per_base["inventory_failures"]
        and not per_base["placement_failures"],
        "all_reflections_engine_valid": not per_base["reflected_engine_failures"],
        "zero_stranded_bases": not per_base["mobility_failures"]
        and not per_base["reflected_mobility_failures"],
        "zero_family_contract_failures": not per_base["family_failures"]
        and not per_base["reflected_family_failures"],
        "zero_exact_duplicates": audit["duplicates"]["exact_duplicate_groups"] == 0,
        "zero_reflection_equivalent_duplicates": audit["duplicates"][
            "reflection_class_duplicate_groups"
        ]
        == 0,
        "zero_stable_id_collisions": audit["duplicates"]["stable_id_collisions"] == 0
        and not audit["duplicates"]["same_id_different_setup"],
        "zero_cross_split_equivalent_leakage": audit["duplicates"][
            "cross_split_class_duplicate_groups"
        ]
        == 0,
        "cross_split_nn_floor_met": similarity["cross_split_min_nn_distance"]
        >= CROSS_SPLIT_FLOOR,
        "all_diversity_thresholds_pass": threshold_result["all_pass"],
        "serialization_exact": not per_base["serialization_failures"]
        and (audit["line_format"] or {}).get("serialization_failures") == 0,
        "reflection_roundtrips_exact": not per_base["reflection_roundtrip_failures"],
        "canonicalization_exact": not per_base["canonicalization_failures"],
        "fingerprints_and_traits_recomputed_exact": not per_base["fingerprint_failures"]
        and not per_base["trait_failures"],
        "identity_split_seed_rederived_exact": not per_base["identity_failures"]
        and not per_base["seed_failures"]
        and not per_base["version_failures"],
        "independent_similarity_agrees": audit["similarity_reconciliation"]["agrees"]
        and similarity["cross_check"]["mismatches_vs_frozen_metric"] == 0,
        "family_self_satisfaction_diagonal_one": not audit["overlap"][
            "diagonal_failures"
        ],
        "manifest_digests_verified": audit["manifest"]["all_pass"],
        "production_library_untouched": library_untouched and manifest_untouched,
    }

    status = "PASS" if all(gates.values()) and audit["status"] == "PASS" else "FAIL"

    payload = {
        "agent": "agent_03",
        "phase": "phase_7",
        "assignment": "Exhaustive Library Audit",
        "status": status,
        "schema_version": "phase_7_agent_03_v1",
        "audit_version": AUDIT_VERSION,
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
            "audit_version": AUDIT_VERSION,
        },
        "library_version": SETUP_LIBRARY_VERSION,
        "contract_version": SETUP_GENERATOR_CONTRACT_VERSION,
        "family_contract_version": SETUP_FAMILY_VERSION,
        "master_seed": DEFAULT_LIBRARY_MASTER_SEED,
        "library_digest": audit["manifest"]["recomputed"]["library_content_digest"],
        "entry_metadata_digest": audit["manifest"]["recomputed"]["entry_metadata_digest"],
        "manifest_digest": audit["manifest"]["recomputed"]["manifest_digest"],
        "setup_count": audit["counts"]["total"],
        "family_counts": audit["counts"]["family_counts"],
        "split_counts": audit["counts"]["split_counts"],
        "family_split_counts": audit["counts"]["family_split_counts"],
        "seeds": {
            "master_seed": DEFAULT_LIBRARY_MASTER_SEED,
            "similarity_cross_check_seed": similarity["cross_check"]["seed"],
        },
        "audit": audit,
        "performance": {
            "audit_wall_seconds": total_seconds,
            "peak_rss_bytes": peak_rss,
            "peak_rss_megabytes": round(peak_rss / 1e6, 1),
            "similarity_method": similarity["method"],
            "ordered_pair_comparisons": similarity["ordered_pair_comparisons"],
            "unordered_pairs": similarity["unordered_pairs"],
            "cell_comparisons": similarity["cell_comparisons"],
            "matrix_seconds": similarity["matrix_seconds"],
            "stage_durations": audit["durations"],
        },
        "tests_before": {
            "command": "python -m pytest -q",
            "passed": 2976,
            "failed": 0,
            "skipped": 3,
            "errors": 0,
            "note": "recorded at commit 974e5e9 before any Agent 3 edit",
        },
        "tests_after": None,
        "commands": [command],
        "durations": {
            "audit_seconds": total_seconds,
            **audit["durations"],
        },
        "files_created": [
            "stratego/setups/audit.py",
            "tests/setups/test_audit.py",
            "scripts/run_phase7_agent03.py",
            "reports/phase_7_data/agent_03_library_audit.json",
            "reports/phase_7_data/agent_03_family_metrics.csv",
            "reports/phase_7_data/agent_03_similarity_audit.csv",
        ],
        "files_modified": [
            "stratego/setups/__init__.py",
            "reports/phase_7_implementation_report.md",
        ],
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "problems": [],
        "deviations": [],
    }

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    AUDIT_ARTIFACT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    family_rows = _write_family_metrics_csv(audit)
    similarity_rows = _write_similarity_csv(audit)
    print(f"family metrics rows: {family_rows}  similarity rows: {similarity_rows}")

    if arguments.run_pytest:
        suite = _run_pytest()
        gates["full_repository_suite_green"] = (
            suite["exit_code"] == 0 and suite["failed"] == 0 and suite["errors"] == 0
        )
        status = "PASS" if all(gates.values()) and audit["status"] == "PASS" else "FAIL"
        payload["tests_after"] = suite
        payload["completion_gates"] = gates
        payload["gates_total"] = len(gates)
        payload["gates_true"] = sum(1 for value in gates.values() if value)
        payload["status"] = status
        AUDIT_ARTIFACT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(
            f"  {'ok ' if gates['full_repository_suite_green'] else 'FAIL'} "
            f"full_repository_suite_green ({suite['summary_line']})"
        )

    print(f"status: {payload['status']}  gates {payload['gates_true']}/{payload['gates_total']}")
    for name, value in gates.items():
        print(f"  {'ok ' if value else 'FAIL'} {name}")
    largest = audit["overlap"]["largest_off_diagonal"]
    print(
        "largest off-diagonal overlap: "
        f"{largest['declared_family']} -> {largest['also_satisfies']} = {largest['fraction']}"
    )
    print(f"audit wall time: {total_seconds}s  peak RSS: {round(peak_rss / 1e6)} MB")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
