#!/usr/bin/env python3
"""Phase 18 Gate G1: the faithful Phase 8 reproduction control.

This is a *driver*, not a reimplementation. Every training decision - the
model, the loss, the example order, the validation cadence, the selection
rule, the resume path - comes from `scripts/run_phase8_agent06.py`, imported
and called here exactly as its own `--full` mode calls it. Nothing in this
file changes Phase 8 semantics.

It exists for two reasons.

**Output isolation.** The accepted harness hard-codes the accepted output
paths. Its `--dry-run` flag redirects every write into the work directory
while leaving the run itself full length, which is what G1 needs: a real
25,000-update run that cannot claim `checkpoints/phase8/warmstart_c1_v1.pt`,
`reports/phase_8_data/` or the Phase 8 report. This driver turns that mode on
and then re-hashes the accepted artifacts afterwards to prove it worked.

**One location assertion that no worktree can satisfy.** Agent 6's
`verify_prerequisites` asserts the corpus root three ways: that the resolver
returns the accepted absolute path, that the pointer file names it, and that
it equals `REPOSITORY_ROOT / <the same path relative to the repository>`. The
third is unsatisfiable outside the original checkout - `REPOSITORY_ROOT` is
wherever the script lives, and G1 is required to run from a clean detached
worktree - and because that function skips the corpus digest verification
entirely when its problem list is non-empty, leaving it to fail would silently
*lose* the 28,000-payload check that G1 depends on.

The override is one line, and it is the narrowest one available: the relative
constant is rebound to the accepted absolute path, so `REPOSITORY_ROOT / ...`
yields that path (an absolute right-hand operand wins in `pathlib`) and the
third assertion collapses onto the first. The first and third assertions still
run unchanged, the corpus is still verified byte by byte, and the Agent 6
gate that reads this - `corpus_resolved_through_resolver` - is computed from
the first assertion and is unaffected either way. The deviation is recorded in
the run artifact rather than hidden; `--prove-location-assertion` demonstrates
the untouched behaviour on demand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_phase8_agent06 as a6  # noqa: E402

RUN_ID = "G1-CONTROL-2026-A"
WORK_PACKAGE = "phase18_setup_integrated_warmstart"

#: Never written, only re-hashed. A change in any of these is a stop condition.
PROTECTED = (
    "checkpoints/phase8/warmstart_c1_v1.pt",
    "checkpoints/phase8/warmstart_c1_v1_manifest.json",
    "checkpoints/phase8/warmstart_c1_v1_initialisation.pt",
    "reports/phase_8_implementation_report.md",
)

LOCATION_ASSERTION = (
    "run_phase8_agent06.verify_prerequisites asserts the resolved corpus root "
    "equals REPOSITORY_ROOT / REQUIRED_CORPUS_ROOT_RELATIVE, which no execution "
    "worktree can satisfy; REQUIRED_CORPUS_ROOT_RELATIVE is rebound to the "
    "accepted absolute REQUIRED_CORPUS_ROOT so the assertion collapses onto the "
    "resolver check that is the actual Phase 8 gate"
)


def log(message: str) -> None:
    print(f"[g1 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_digests(root: Path) -> dict:
    return {name: sha256(root / name) for name in PROTECTED}


def apply_location_override() -> dict:
    """Rebind the one location constant, and record exactly what changed."""
    before = a6.REQUIRED_CORPUS_ROOT_RELATIVE
    a6.REQUIRED_CORPUS_ROOT_RELATIVE = a6.REQUIRED_CORPUS_ROOT
    return {
        "declared_deviation": LOCATION_ASSERTION,
        "constant": "run_phase8_agent06.REQUIRED_CORPUS_ROOT_RELATIVE",
        "before": before,
        "after": a6.REQUIRED_CORPUS_ROOT_RELATIVE,
        "assertions_left_untouched": [
            "str(default_corpus_root()) == REQUIRED_CORPUS_ROOT",
            "describe_corpus_root()['pointer_value'] == REQUIRED_CORPUS_ROOT",
            "run_segment's own REQUIRED_CORPUS_ROOT check",
            "verify_corpus_identity over all 28,000 payloads",
        ],
        "phase8_semantics_changed": "none",
    }


def prove_location_assertion() -> dict:
    """Run the untouched assertion and show it is the only thing that fails."""
    from stratego.training import synthetic_corpus as sc

    resolved = sc.default_corpus_root()
    required_via_repository = a6.REPOSITORY_ROOT / a6.REQUIRED_CORPUS_ROOT_RELATIVE
    return {
        "resolved": str(resolved),
        "required_corpus_root": a6.REQUIRED_CORPUS_ROOT,
        "repository_root": str(a6.REPOSITORY_ROOT),
        "required_via_repository": str(required_via_repository),
        "resolver_assertion_holds": str(resolved) == a6.REQUIRED_CORPUS_ROOT,
        "pointer_assertion_holds": (
            sc.describe_corpus_root()["pointer_value"] == a6.REQUIRED_CORPUS_ROOT
        ),
        "repository_relative_assertion_holds": resolved == required_via_repository,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run-pytest", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--prove-location-assertion", action="store_true")
    parser.add_argument("--skip-payload-bytes", action="store_true")
    arguments = parser.parse_args()

    if arguments.prove_location_assertion:
        print(json.dumps(prove_location_assertion(), indent=2, sort_keys=True))
        return 0
    if not arguments.work_dir:
        parser.error("--work-dir is required for every mode except the proof")

    work = Path(arguments.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    # Output isolation first, so nothing downstream can address an accepted path.
    a6.configure_output(dry_run=True, work=work)
    override = apply_location_override()
    log(f"outputs isolated under {work}")
    log("location override applied; corpus assertions otherwise untouched")

    before = protected_digests(a6.REPOSITORY_ROOT)
    started = time.perf_counter()

    log("verifying prerequisites, identities and all 28,000 corpus payloads")
    verification, _identity = a6.verify_prerequisites(
        check_payload_bytes=not arguments.skip_payload_bytes,
        device=arguments.device,
    )
    a6.write_json(work / "verification.json", verification)
    if verification["problems"]:
        for problem in verification["problems"]:
            log(f"BLOCKED: {problem}")
        return 1
    log("prerequisites verified")

    if arguments.verify_only:
        a6.write_json(
            work / "phase18_g1_preflight.json",
            {
                "artifact": "phase18_g1_preflight",
                "run_id": RUN_ID,
                "work_package": WORK_PACKAGE,
                "location_override": override,
                "location_assertion_proof": prove_location_assertion(),
                "verification": verification,
                "protected_before": before,
                "protected_after": protected_digests(a6.REPOSITORY_ROOT),
                "seconds": round(time.perf_counter() - started, 3),
            },
        )
        log("verify-only complete")
        return 0

    tests: dict = {}
    tests_path = work / "tests.json"
    if tests_path.exists():
        tests = json.loads(tests_path.read_text())
    if arguments.run_pytest and "before" not in tests:
        log("running the full suite before the control")
        tests["before"] = a6.run_pytest()
        log(f"suite before: {tests['before']['summary']}")
        a6.write_json(tests_path, tests)
        if tests["before"]["returncode"] != 0:
            log("BLOCKED: the suite is not green before the run")
            return 1

    frozen = a6.frozen_config_payload()
    updates = int(frozen["config"]["max_final_updates"])
    restart_at = a6.DEFAULT_RESTART_AT
    topology = a6.LoaderTopology(**frozen["config"]["loader_topology"])

    run_path = work / "canonical_run.json"
    if run_path.exists():
        log("reusing the completed canonical run in the work directory")
        run = json.loads(run_path.read_text())
    else:
        log(f"canonical run: {updates:,} updates, restart at {restart_at:,}")
        run = a6.run_canonical(
            work=work,
            device=arguments.device,
            updates=updates,
            restart_at=restart_at,
            topology=topology,
        )

    log("freezing the best validation checkpoint into the Phase 18 work directory")
    manifest = a6.freeze_checkpoint(
        work=work,
        run=run,
        device=arguments.device,
        topology=topology,
        full_validation=True,
    )
    log(f"frozen: step {manifest['selected_global_step']} sha256 {manifest['checkpoint_sha256']}")

    curve_rows = a6.assemble_curve(work, run)
    a6.write_curve(curve_rows, a6.artifact_path("agent_06_training_curve.csv"))

    if arguments.run_pytest and "after" not in tests:
        log("running the full suite after the control")
        tests["after"] = a6.run_pytest()
        log(f"suite after: {tests['after']['summary']}")
        a6.write_json(tests_path, tests)

    artifact = a6.build_run_artifact(
        verification=verification,
        run=run,
        manifest=manifest,
        curve_rows=curve_rows,
        tests=tests,
    )
    a6.write_json(a6.artifact_path("agent_06_warmstart_run.json"), artifact)
    a6.write_json(a6.artifact_path("agent_06_checkpoint_manifest.json"), manifest)

    after = protected_digests(a6.REPOSITORY_ROOT)
    unchanged = before == after

    summary = {
        "artifact": "phase18_g1_control_run",
        "run_id": RUN_ID,
        "work_package": WORK_PACKAGE,
        "gate": "G1",
        "work_directory": str(work),
        "output_isolation": {
            "mode": "run_phase8_agent06 --dry-run (output isolation only)",
            "output_directory": str(a6._OUTPUT_DIRECTORY),
            "checkpoint_path": str(a6._CHECKPOINT_PATH),
            "manifest_path": str(a6._MANIFEST_PATH),
            "phase8_report_appended": a6._APPEND_REPORT,
            "updates_planned": updates,
            "shortened": False,
        },
        "location_override": override,
        "location_assertion_proof": prove_location_assertion(),
        "accepted_artifacts": {
            "before": before,
            "after": after,
            "unchanged": unchanged,
        },
        "agent06_status": artifact["status"],
        "agent06_completion_gates": artifact["completion_gates"],
        "selected_global_step": manifest["selected_global_step"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "tests": tests,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    a6.write_json(work / "phase18_g1_control_summary.json", summary)

    if not unchanged:
        log("BLOCKED: an accepted Phase 8 artifact changed during the control")
        return 1
    failing = [name for name, ok in artifact["completion_gates"].items() if not ok]
    if failing:
        log(f"Agent 6 gates failing: {failing}")
        return 1
    log(f"control complete: {artifact['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
