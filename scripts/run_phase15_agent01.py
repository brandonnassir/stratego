#!/usr/bin/env python3
"""Phase 15 Agent 1 — clean belief-corpus generation and B18/B24 fine-tuning.

Specification source:
`instructions/phase_15_belief_search_engineering/01_AGENT_1_BELIEF_HEAD_TRAINING.md`

This script controls nothing outside Phase 15. It never creates an
emergency-stop file, sends a signal, edits Phase 14 run state, rotates a
live checkpoint or invokes a closeout command. Its only reads of Phase 14
are the candidate ledger, the two candidate evaluation weight files and the
two archive snapshots they name — all opened read-only.

Roles, in the order they must run:

```text
boundary      inspect live process/status state, read-only
orientation   the section 4 gate; nothing else may run until it passes
sources       resolve and freeze P18/P24 from the Phase 14 ledger
corpus        generate phase15_belief_corpus_v1
verify        split-overlap, mixture and label checks on the stored bytes
train         fine-tune B18 and B24 under the one declared recipe
calibrate     fit one temperature per specialist on the calibration split
metrics       the full section 11 metric block on development
interface     the section 12 belief/sampler provider checks
handoff       write phase15_search_handoff_v1.json
report        write the Agent 1 report and summary
```
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.belief.phase15.contract import (  # noqa: E402
    CORPUS_SPLITS,
    PHASE15_STATUS_MARKERS,
    POSITION_TARGET,
    RECIPE,
)

DATA_ROOT = REPOSITORY_ROOT / "data" / "phase15" / "phase15_belief_corpus_v1"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "phase15"
REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase15"

SOURCE_CHECKPOINTS = {
    "p18": str(CHECKPOINT_ROOT / "p18_source_readonly.pt"),
    "p24": str(CHECKPOINT_ROOT / "p24_source_readonly.pt"),
}


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _read(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing prerequisite artifact: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1. Process boundary
# ---------------------------------------------------------------------------


def role_boundary(args) -> dict:
    """Inspect the live process/status state. Read-only, and no signals."""
    processes = subprocess.run(
        ["ps", "-Ao", "pid,pcpu,etime,command"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    interesting = [
        line.strip()
        for line in processes
        if ("phase14" in line or "phase15" in line or "stratego" in line)
        and "grep" not in line
        and "run_phase15_agent01" not in line
    ]
    competing = [
        line
        for line in interesting
        if any(
            token in line
            for token in (
                "phase14_launch",
                "phase14_runner",
                "phase14_trainer",
                "phase14_collector",
                "phase14_evaluate",
                "phase14_supervisor",
            )
        )
    ]
    run_root = Path(args.phase14_root)
    state_path = run_root / "phase14_run_state.json"
    state: dict = {}
    if state_path.is_file():
        raw = json.loads(state_path.read_text())
        progress = raw.get("progress", {})
        state = {
            "path": str(state_path),
            "elapsed_hours": raw.get("elapsed_hours"),
            "closed": progress.get("closed"),
            "closed_reason": raw.get("closed_reason"),
            "iterations_completed": progress.get("iterations_completed"),
            "last_candidate_index": progress.get("last_candidate_index"),
            "mtime_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(state_path.stat().st_mtime)
            ),
        }
    verdict = "ready_for_compute" if not competing else "blocked_phase14_is_running"
    block = {
        "artifact": "phase15_agent01_process_boundary_v1",
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase15_authorizes_process_control": False,
        "actions_taken": "none — this role reads process and status state only",
        "matching_processes": interesting,
        "competing_phase14_processes": competing,
        "phase14_run_state": state,
        "host_cpus": os.cpu_count(),
        "verdict": verdict,
        **PHASE15_STATUS_MARKERS,
    }
    _write(REPORT_ROOT / "agent_01_process_boundary.json", block)
    return block


# ---------------------------------------------------------------------------
# 2. Orientation gate
# ---------------------------------------------------------------------------


def role_orientation(args) -> dict:
    from stratego.belief.phase15.orientation import orientation_gate

    evidence = orientation_gate(boards=int(args.orientation_boards))
    _write(REPORT_ROOT / "agent_01_orientation_gate.json", evidence)
    return evidence


# ---------------------------------------------------------------------------
# 3. Freeze P18 / P24
# ---------------------------------------------------------------------------


def role_sources(args) -> dict:
    from stratego.belief.phase15.sources import freeze_sources

    frozen = freeze_sources(
        run_root=args.phase14_root, destination_root=CHECKPOINT_ROOT
    )
    return {source_id: source.to_dict() for source_id, source in frozen.items()}


# ---------------------------------------------------------------------------
# 4. Corpus
# ---------------------------------------------------------------------------


def role_corpus(args) -> dict:
    from stratego.belief.phase15.build import build_corpus

    evidence = _read(REPORT_ROOT / "agent_01_orientation_gate.json")
    identity = {
        source_id: _read(CHECKPOINT_ROOT / f"{source_id}_source_identity.json")
        for source_id in SOURCE_CHECKPOINTS
    }
    targets = {split: POSITION_TARGET[split] for split in CORPUS_SPLITS}
    if args.positions:
        targets = {split: int(args.positions) for split in CORPUS_SPLITS}

    started = time.perf_counter()

    def progress(split, samples, target, seconds):
        rate = samples / seconds if seconds > 0 else 0.0
        print(
            f"[{time.strftime('%H:%M:%S')}] {split}: {samples}/{target} positions "
            f"({rate:.1f}/s, {seconds / 60:.1f} min)",
            flush=True,
        )

    manifest = build_corpus(
        DATA_ROOT,
        SOURCE_CHECKPOINTS,
        identity,
        evidence,
        targets=targets,
        workers=int(args.workers),
        overwrite=bool(args.overwrite),
        progress=progress,
    )
    print(
        f"corpus complete in {(time.perf_counter() - started) / 60:.1f} min: "
        f"{manifest['corpus_digest'][:16]}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# 5. Verify the stored corpus
# ---------------------------------------------------------------------------


def role_verify(args) -> dict:
    from stratego.belief.phase15.verify import verify_corpus

    report = verify_corpus(DATA_ROOT, orientation_games=int(args.orientation_games))
    _write(REPORT_ROOT / "agent_01_corpus_verification.json", report)
    return report


# ---------------------------------------------------------------------------
# 6-8. Train, calibrate, evaluate
# ---------------------------------------------------------------------------


def role_train(args) -> dict:
    import numpy as np

    from stratego.belief.phase15.checkpoint import save_specialist
    from stratego.belief.phase15.contract import SPECIALISTS
    from stratego.belief.phase15.metrics import baseline_probabilities
    from stratego.belief.phase15.pipeline import (
        agent1c_reference,
        comparison_block,
        load_corpus,
        train_one,
    )
    from stratego.belief.phase15.storage import read_manifest

    manifest = read_manifest(DATA_ROOT)
    identity = {
        source_id: _read(CHECKPOINT_ROOT / f"{source_id}_source_identity.json")
        for source_id in SOURCE_CHECKPOINTS
    }
    splits = load_corpus(DATA_ROOT)

    def progress(*payload):
        if len(payload) == 2 and isinstance(payload[1], dict):
            specialist, row = payload
            print(
                f"[{time.strftime('%H:%M:%S')}] {specialist} epoch {row['epoch']:>2}: "
                f"train {row['train_loss']:.4f}  dev CE {row['dev_ce']:.4f}  "
                f"R_CE {row['dev_r_ce']:.4f}  top1 {row['dev_top1']:.4f}  "
                f"({row['seconds'] / 60:.1f} min)",
                flush=True,
            )

    trained = {}
    curves = {}
    for specialist_id in SPECIALISTS:
        source_id = specialist_id.replace("b", "p")
        print(f"[{time.strftime('%H:%M:%S')}] training {specialist_id}", flush=True)
        result = train_one(
            specialist_id,
            identity[source_id],
            splits,
            CHECKPOINT_ROOT,
            device=args.device,
            batch_size=args.batch_size or None,
            batch_size_reason=args.batch_size_reason or None,
            rebuild_caches=bool(args.rebuild_caches),
            progress=progress,
        )
        trained[specialist_id] = result
        curves[specialist_id] = result["training"]["curve"]

    development = splits["development"]
    dev_baseline = baseline_probabilities(development)
    reference = agent1c_reference(development, device=args.device)
    comparison = comparison_block(trained, reference, development, dev_baseline)

    saved = {}
    for specialist_id, result in trained.items():
        block = save_specialist(
            result["model"],
            CHECKPOINT_ROOT / f"{specialist_id}_belief_v1.pt",
            source_identity=identity[result["source_id"]],
            corpus_identity=manifest,
            training_record=result["training"],
            calibration_record=result["calibration"],
            overwrite=bool(args.overwrite),
        )
        block["calibration"] = result["calibration"]
        saved[specialist_id] = block
        np.save(
            CHECKPOINT_ROOT / f"{specialist_id}_development_probabilities.npy",
            result["development_probabilities"],
        )

    metrics = {
        "artifact": "phase15_agent01_metrics_v1",
        "corpus_digest": manifest["corpus_digest"],
        "comparison": comparison,
        "specialists": {
            specialist_id: {
                "source_id": result["source_id"],
                "training": {
                    key: value
                    for key, value in result["training"].items()
                    if key != "curve"
                },
                "calibration": result["calibration"],
                "development_raw": result["development_raw"],
                "development_calibrated": result["development_calibrated"],
                "checkpoint": saved[specialist_id],
                "caches": result["caches"],
            }
            for specialist_id, result in trained.items()
        },
        **PHASE15_STATUS_MARKERS,
    }
    _write(REPORT_ROOT / "agent_01_metrics.json", metrics)
    _write(
        REPORT_ROOT / "agent_01_learning_curves.json",
        {
            "artifact": "phase15_agent01_learning_curves_v1",
            "curves": curves,
            "recipe": dict(RECIPE),
            **PHASE15_STATUS_MARKERS,
        },
    )
    return {
        specialist_id: {
            "checkpoint": saved[specialist_id]["path"],
            "sha256": saved[specialist_id]["sha256"],
            "temperature": saved[specialist_id]["temperature"],
            "best_epoch": trained[specialist_id]["training"]["best_epoch"],
            "development_r_ce": comparison["specialists"][specialist_id]["r_ce"],
        }
        for specialist_id in saved
    }


# ---------------------------------------------------------------------------
# 9. The provider checks
# ---------------------------------------------------------------------------


def role_interface(args) -> dict:
    from stratego.belief.phase15.checkpoint import load_specialist
    from stratego.belief.phase15.contract import SPECIALISTS
    from stratego.belief.phase15.interface import Phase15BeliefProvider
    from stratego.belief.phase15.interface_checks import (
        check_provider,
        check_truth_isolation,
        collect_check_positions,
    )
    from stratego.belief.phase15.pipeline import load_policy
    from stratego.belief.phase15.setups import Phase15SetupSources
    from stratego.evaluation.neural_worker import DECISION_MODE_GREEDY, InferenceOwner
    from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config

    owners = {
        source_id: InferenceOwner(
            path,
            decision_mode=DECISION_MODE_GREEDY,
            device="cpu",
            dtype="float32",
            expected_architecture_id=ARCHITECTURE_FAMILY,
            expected_configuration=candidate_config("C1"),
            name=f"phase15_{source_id}",
        )
        for source_id, path in SOURCE_CHECKPOINTS.items()
    }
    positions = collect_check_positions(
        owners, Phase15SetupSources(), games=int(args.check_games), per_game=4
    )
    print(f"collected {len(positions)} check positions", flush=True)

    reports = {}
    for specialist_id in SPECIALISTS:
        source_id = specialist_id.replace("b", "p")
        policy_model, _metadata = load_policy(SOURCE_CHECKPOINTS[source_id])
        specialist, payload = load_specialist(
            CHECKPOINT_ROOT / f"{specialist_id}_belief_v1.pt", policy_model
        )
        provider = Phase15BeliefProvider(
            policy_model,
            specialist,
            provider_id=f"phase15_{specialist_id}",
            identity={
                "specialist_id": specialist_id,
                "bound_policy": source_id,
                "state_digest": payload["state_digest"],
            },
            calibrated=payload["calibration"].get("keep_calibrated", False),
        )
        report = check_provider(provider, positions, worlds=int(args.check_worlds))
        report["truth_isolation"] = check_truth_isolation(provider, positions)
        report["describe"] = provider.describe()
        reports[specialist_id] = report
        print(
            f"{specialist_id}: {report['positions_checked']} positions, "
            f"{report['worlds_checked']} worlds, all checks passed",
            flush=True,
        )
    block = {
        "artifact": "phase15_agent01_interface_checks_v1",
        "providers": reports,
        **PHASE15_STATUS_MARKERS,
    }
    _write(REPORT_ROOT / "agent_01_interface_checks.json", block)
    return block


# ---------------------------------------------------------------------------
# 10. The handoff
# ---------------------------------------------------------------------------


def role_handoff(args) -> dict:
    from stratego.belief.phase15.handoff import (
        build_handoff,
        verify_handoff,
        write_handoff,
    )
    from stratego.belief.phase15.storage import read_manifest

    manifest = read_manifest(DATA_ROOT)
    metrics = _read(REPORT_ROOT / "agent_01_metrics.json")
    checks = _read(REPORT_ROOT / "agent_01_interface_checks.json")
    sources = {
        source_id: _read(CHECKPOINT_ROOT / f"{source_id}_source_identity.json")
        for source_id in SOURCE_CHECKPOINTS
    }
    specialists = {
        specialist_id: {
            **block["checkpoint"],
            "calibration": block["calibration"],
        }
        for specialist_id, block in metrics["specialists"].items()
    }
    document = build_handoff(
        sources=sources,
        specialists=specialists,
        corpus_manifest=manifest,
        interface_reports={
            specialist_id: {
                key: report[key]
                for key in (
                    "positions_checked",
                    "worlds_checked",
                    "probabilities_sum_to_one",
                    "fixed_seed_reproduces_worlds",
                    "remaining_piece_counts_exact",
                    "moved_pieces_never_immobile",
                    "all_worlds_pass_accepted_validation",
                    "marginal_latency_ms",
                    "passed",
                )
            }
            for specialist_id, report in checks["providers"].items()
        },
        development_metrics=metrics["comparison"],
        root=REPOSITORY_ROOT,
    )
    document["verification"] = verify_handoff(document, root=REPOSITORY_ROOT)
    if not document["verification"]["verified"]:
        raise SystemExit(
            f"the handoff does not verify: {document['verification']['findings']}"
        )
    write_handoff(document, REPORT_ROOT / "phase15_search_handoff_v1.json")
    return document


# ---------------------------------------------------------------------------
# 11. The report
# ---------------------------------------------------------------------------


def _table(rows, headers) -> str:
    widths = [
        max(len(str(headers[column])), *(len(str(row[column])) for row in rows))
        for column in range(len(headers))
    ]
    def line(values):
        return "| " + " | ".join(
            str(value).ljust(widths[column]) for column, value in enumerate(values)
        ) + " |"
    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def role_report(args) -> dict:
    from stratego.belief.phase15.contract import SPECIALISTS
    from stratego.belief.phase15.storage import read_manifest

    manifest = read_manifest(DATA_ROOT)
    metrics = _read(REPORT_ROOT / "agent_01_metrics.json")
    checks = _read(REPORT_ROOT / "agent_01_interface_checks.json")
    verification = _read(REPORT_ROOT / "agent_01_corpus_verification.json")
    orientation = _read(REPORT_ROOT / "agent_01_orientation_gate.json")
    boundary = _read(REPORT_ROOT / "agent_01_process_boundary.json")
    handoff = _read(REPORT_ROOT / "phase15_search_handoff_v1.json")
    curves = _read(REPORT_ROOT / "agent_01_learning_curves.json")
    sources = {
        source_id: _read(CHECKPOINT_ROOT / f"{source_id}_source_identity.json")
        for source_id in SOURCE_CHECKPOINTS
    }
    comparison = metrics["comparison"]

    summary = {
        "artifact": "phase15_agent01_summary_v1",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **PHASE15_STATUS_MARKERS,
        "process_boundary": boundary["verdict"],
        "orientation_gate": {
            "passed": orientation["passed"],
            "armies_checked": orientation["armies_checked"],
            "front_row_flag_rate": orientation["front_row_flag_rate"],
            "defect_counterfactual_rate": orientation["defect_counterfactual"]["rate"],
            "negative_canary_detected": orientation["negative_canary"]["detected"],
        },
        "corpus": {
            "corpus_digest": manifest["corpus_digest"],
            "positions": {
                split: block["samples"]
                for split, block in sorted(manifest["splits"].items())
            },
            "targets_met": all(
                block["met_target"] for block in manifest["splits"].values()
            ),
            "pieces": sum(
                block["pieces"] for block in manifest["splits"].values()
            ),
            "generation_minutes": round(
                sum(manifest["generation_seconds"].values()) / 60.0, 2
            ),
            "verified": verification["passed"],
            "splits_disjoint": verification["disjointness"]["disjoint"],
        },
        "policy_sources": {
            source_id: {
                "logical_identity": block["logical_identity"],
                "model_state_digest": block["model_state_digest"],
                "unchanged_by_training": metrics["specialists"][
                    source_id.replace("p", "b")
                ]["training"]["source_unchanged"]["unchanged"],
            }
            for source_id, block in sources.items()
        },
        "specialists": {
            specialist_id: {
                "checkpoint_sha256": metrics["specialists"][specialist_id][
                    "checkpoint"
                ]["sha256"],
                "best_epoch": metrics["specialists"][specialist_id]["training"][
                    "best_epoch"
                ],
                "epochs_run": metrics["specialists"][specialist_id]["training"][
                    "epochs_run"
                ],
                "stopped_because": metrics["specialists"][specialist_id]["training"][
                    "stopped_because"
                ],
                "temperature": metrics["specialists"][specialist_id]["calibration"][
                    "applied_temperature"
                ],
                "keep_calibrated": metrics["specialists"][specialist_id][
                    "calibration"
                ]["keep_calibrated"],
                **comparison["specialists"][specialist_id],
                "interface_checks_passed": checks["providers"][specialist_id][
                    "passed"
                ],
            }
            for specialist_id in SPECIALISTS
        },
        "agent1c_reference": comparison["agent1c_reference"],
        "handoff": {
            "artifact": handoff["artifact"],
            "verified": handoff["verification"]["verified"],
            "path": str(
                (REPORT_ROOT / "phase15_search_handoff_v1.json").relative_to(
                    REPOSITORY_ROOT
                )
            ),
        },
    }
    _write(REPORT_ROOT / "agent_01_summary.json", summary)

    specialist_rows = [
        [
            specialist_id.upper(),
            f"P{sources[SPECIALIST_OF[specialist_id]]['hour']:02d}",
            f"{comparison['specialists'][specialist_id]['ce']:.4f}",
            f"{comparison['specialists'][specialist_id]['r_ce']:.4f}",
            "[{:.4f}, {:.4f}]".format(
                *comparison["specialists"][specialist_id]["r_ce_ci95"]
            ),
            f"{comparison['specialists'][specialist_id]['top1']:.4f}",
            f"{comparison['specialists'][specialist_id]['brier']:.4f}",
            f"{comparison['specialists'][specialist_id]['expected_calibration_error']:.4f}",
        ]
        for specialist_id in SPECIALISTS
    ]
    specialist_rows.append(
        [
            "Agent 1C",
            "Phase 9 C1",
            f"{comparison['agent1c_reference']['ce']:.4f}",
            f"{comparison['agent1c_reference']['r_ce']:.4f}",
            "[{:.4f}, {:.4f}]".format(*comparison["agent1c_reference"]["r_ce_ci95"]),
            f"{comparison['agent1c_reference']['top1']:.4f}",
            f"{comparison['agent1c_reference']['brier']:.4f}",
            f"{comparison['agent1c_reference']['expected_calibration_error']:.4f}",
        ]
    )
    specialist_rows.append(
        [
            "count baseline",
            "—",
            f"{comparison['remaining_count_baseline']['ce']:.4f}",
            "1.0000",
            "—",
            f"{comparison['remaining_count_baseline']['top1']:.4f}",
            f"{comparison['remaining_count_baseline']['brier']:.4f}",
            "—",
        ]
    )

    document = _render_report(
        manifest=manifest,
        metrics=metrics,
        checks=checks,
        verification=verification,
        orientation=orientation,
        boundary=boundary,
        handoff=handoff,
        curves=curves,
        sources=sources,
        summary=summary,
        specialist_table=_table(
            specialist_rows,
            ["model", "backbone", "CE", "R_CE", "R_CE 95% CI", "top-1", "Brier", "ECE"],
        ),
    )
    path = REPORT_ROOT / "agent_01_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
    return {"report": str(path), "summary": summary}


SPECIALIST_OF = {"b18": "p18", "b24": "p24"}


def _render_report(**block) -> str:
    from stratego.belief.phase15.report_text import render

    return render(**block)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ROLES = {
    "boundary": role_boundary,
    "orientation": role_orientation,
    "sources": role_sources,
    "corpus": role_corpus,
    "verify": role_verify,
    "train": role_train,
    "interface": role_interface,
    "handoff": role_handoff,
    "report": role_report,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument(
        "--phase14-root",
        default="/Volumes/Brandon_Washington/stratego_phase14",
        help="Phase 14 run root. Opened read-only.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--orientation-boards", type=int, default=4096)
    parser.add_argument(
        "--positions",
        type=int,
        default=0,
        help="override every split's position budget (pilot use only)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batch-size-reason", default="")
    parser.add_argument("--rebuild-caches", action="store_true")
    parser.add_argument("--orientation-games", type=int, default=256)
    parser.add_argument("--check-games", type=int, default=8)
    parser.add_argument("--check-worlds", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = ROLES[args.role](args)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
