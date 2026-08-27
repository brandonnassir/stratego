#!/usr/bin/env python
"""Phase 16 Agent 1 orchestration: build, score, analyse, hand off.

Roles
-----
```text
build      author the adversarial library; freeze both pack manifests
bench      score one arm on phase16_benchmark_v1        (heavy: lock held)
baseline   run the adversarial baseline arms at a preset (heavy: lock held)
analyse    read the result files and write the two baseline JSON reports
handoff    re-verify every digest and write phase16_measurement_handoff_v1
backup     refresh the untracked backup (new dated file) and record it
```

Heavy roles honour `checkpoints/phase16/COMPUTE_LOCK.json`: they refuse to
start while another agent's live pid holds the lock, create it for the run,
and delete it on exit. Packs append rows to a JSONL as games finish, so a
killed run resumes where it stopped by re-running the identical command.

Examples:

    .venv/bin/python scripts/run_phase16_agent01.py --role build
    .venv/bin/python scripts/run_phase16_agent01.py --role bench --arm p24_direct
    .venv/bin/python scripts/run_phase16_agent01.py --role bench --arm p24_b24 --preset TINY
    .venv/bin/python scripts/run_phase16_agent01.py --role baseline --preset TINY \
        --arms benchmark_control,adversarial_opponent,adversarial_both
    .venv/bin/python scripts/run_phase16_agent01.py --role analyse
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

LOCK_PATH = REPOSITORY_ROOT / "checkpoints/phase16/COMPUTE_LOCK.json"
PACK_DIR = REPOSITORY_ROOT / "reports/phase16/packs"
BENCH_SUMMARY_PATH = REPOSITORY_ROOT / "reports/phase16/agent_01_benchmark_baselines.json"
BASELINE_SUMMARY_PATH = REPOSITORY_ROOT / "reports/phase16/agent_01_adversarial_baseline.json"
HANDOFF_PATH = REPOSITORY_ROOT / "reports/phase16/phase16_measurement_handoff_v1.json"
BACKUP_RECORD_PATH = REPOSITORY_ROOT / "reports/phase16/agent_01_backup.json"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


# ---------------------------------------------------------------------------
# The compute lock (overview section 5)
# ---------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(task: str, expected_hours: float) -> None:
    announced = False
    while LOCK_PATH.is_file():
        held = json.loads(LOCK_PATH.read_text())
        holder = held.get("pid")
        if holder is None or not pid_alive(holder) or int(holder) == os.getpid():
            log(f"stale compute lock (pid {holder} not alive) — taking over")
            break
        if not announced:
            log(
                f"COMPUTE_LOCK held by agent {held.get('agent')} pid {holder} "
                f"({held.get('task')}); waiting — never co-running heavy compute"
            )
            announced = True
        time.sleep(60)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        json.dumps(
            {
                "agent": 1,
                "task": task,
                "started_utc": utc_now(),
                "expected_hours": expected_hours,
                "pid": os.getpid(),
            },
            indent=1,
        )
        + "\n"
    )
    log(f"compute lock acquired: {task}")


def release_lock() -> None:
    if LOCK_PATH.is_file():
        held = json.loads(LOCK_PATH.read_text())
        if held.get("pid") == os.getpid():
            LOCK_PATH.unlink()
            log("compute lock released")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def role_build(arguments) -> int:
    from stratego.evaluation.phase16 import adversarial, baseline, benchmark
    from stratego.search.phase15.boards import Phase15MatchSetupSources

    root = REPOSITORY_ROOT
    generated = utc_now()

    library_path = root / adversarial.DEFAULT_LIBRARY_PATH
    if library_path.is_file() and not arguments.rebuild:
        document = adversarial.load_library(root=root)
        log(f"adversarial library already frozen ({document['library_digest'][:16]}…)")
    else:
        log("authoring the adversarial library…")
        document = adversarial.build_library_document(
            adversarial.author_library(), generated_utc=generated
        )
        adversarial.save_library(document, root=root)
        log(
            f"library: {document['setup_count']} setups, authored digest "
            f"{document['authored_digest'][:16]}…"
        )

    sources = Phase15MatchSetupSources()

    bench_path = root / benchmark.DEFAULT_MANIFEST_PATH
    if bench_path.is_file() and not arguments.rebuild:
        manifest = benchmark.load_benchmark_manifest(root=root)
        log(f"benchmark manifest already frozen ({manifest['manifest_digest'][:16]}…)")
    else:
        log("drawing the 120 benchmark boards…")
        plans = benchmark.benchmark_plans(sources=sources)
        manifest = benchmark.build_benchmark_manifest(
            plans, generated_utc=generated, sources=sources
        )
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        bench_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        log(
            f"benchmark: {manifest['board_count']} boards, digest "
            f"{manifest['manifest_digest'][:16]}…"
        )

    base_path = root / baseline.DEFAULT_MANIFEST_PATH
    if base_path.is_file() and not arguments.rebuild:
        base_manifest = baseline.load_baseline_manifest(root=root)
        log(f"baseline manifest already frozen ({base_manifest['manifest_digest'][:16]}…)")
    else:
        log("drawing the 288 baseline arm boards…")
        plans = baseline.baseline_plans(library=document, sources=sources, root=root)
        base_manifest = baseline.build_baseline_manifest(
            plans, generated_utc=generated, library_digest=document["library_digest"]
        )
        base_path.write_text(json.dumps(base_manifest, indent=1, sort_keys=True) + "\n")
        log(
            f"baseline: {base_manifest['board_count']} boards, digest "
            f"{base_manifest['manifest_digest'][:16]}…"
        )

    log("verifying both manifests rebuild from identity alone…")
    benchmark.materialize_benchmark(
        benchmark.load_benchmark_manifest(root=root), sources=sources, verify=True
    )
    baseline.materialize_baseline(
        baseline.load_baseline_manifest(root=root),
        library=adversarial.load_library(root=root),
        sources=sources,
        root=root,
        verify=True,
    )
    log("build complete; both manifests verified")
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def _progress(done: int, total: int, result: dict) -> None:
    row = result["row"]
    log(
        f"  {done}/{total} {row['board_id'].split('|', 2)[-1]} -> "
        f"{row['outcome']} ({row['plies']} plies, {row['wall_seconds']:.0f}s)"
    )


def role_bench(arguments) -> int:
    from stratego.evaluation.phase16.runner import score_on_benchmark

    arm = arguments.arm
    preset = arguments.preset
    label = f"{arm}_{'direct' if arm.endswith('_direct') else preset}"
    subset = arguments.subset if arguments.subset else None
    if subset:
        label += f"_{subset}"
    out_path = PACK_DIR / f"benchmark_{label}.jsonl"
    summary_path = PACK_DIR / f"benchmark_{label}_summary.json"
    acquire_lock(f"agent1 benchmark pack {label}", arguments.expected_hours)
    try:
        started = time.time()
        report = score_on_benchmark(
            arm,
            preset=preset,
            workers=arguments.workers,
            subset=subset,
            root=str(REPOSITORY_ROOT),
            device=arguments.device,
            out_path=out_path,
            progress=_progress if not arguments.quiet else None,
        )
        report["minutes"] = round((time.time() - started) / 60, 1)
        report["workers"] = arguments.workers
        summary = dict(report)
        summary.pop("rows")
        summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        log(
            f"bench {label}: EWR {report['summary']['ewr']} over "
            f"{report['games']} games in {report['minutes']} min -> {summary_path}"
        )
    finally:
        release_lock()
    return 0


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


def role_baseline(arguments) -> int:
    from stratego.evaluation.phase16 import baseline
    from stratego.evaluation.phase16.contract import BASELINE_PAIRING
    from stratego.evaluation.phase16.runner import Task16, normalize_seat_spec, run_pack16

    manifest = baseline.load_baseline_manifest(root=REPOSITORY_ROOT)
    arms = tuple(arguments.arms.split(","))
    preset = arguments.preset
    spec = normalize_seat_spec(BASELINE_PAIRING)
    boards = [
        row["board_id"]
        for row in manifest["boards"]
        if row["setup_source"] in arms
    ]
    out_path = PACK_DIR / f"adversarial_{preset}.jsonl"
    acquire_lock(
        f"agent1 adversarial baseline {preset} ({','.join(arms)})",
        arguments.expected_hours,
    )
    try:
        started = time.time()
        results = run_pack16(
            [Task16(spec, preset, board) for board in boards],
            root=str(REPOSITORY_ROOT),
            device=arguments.device,
            workers=arguments.workers,
            out_path=out_path,
            progress=_progress if not arguments.quiet else None,
        )
        minutes = round((time.time() - started) / 60, 1)
        log(f"baseline {preset}: {len(results)} games in {minutes} min -> {out_path}")
    finally:
        release_lock()
    return 0


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> "list[dict]":
    from stratego.evaluation.phase16.runner import load_results

    return [entry["row"] for entry in load_results(path).values()]


def role_analyse(arguments) -> int:
    from stratego.evaluation.phase16.analysis import analyse_baseline
    from stratego.evaluation.phase16.benchmark import load_benchmark_manifest
    from stratego.evaluation.phase16.contract import PHASE16_STATUS_MARKERS

    benchmark_manifest = load_benchmark_manifest(root=REPOSITORY_ROOT)
    bench = {
        "artifact": "phase16_agent01_benchmark_baselines_v1",
        **PHASE16_STATUS_MARKERS,
        "written_utc": utc_now(),
        "pack": benchmark_manifest["artifact"],
        "manifest_digest": benchmark_manifest["manifest_digest"],
        "baselines": {},
    }
    for summary_file in sorted(PACK_DIR.glob("benchmark_*_summary.json")):
        summary = json.loads(summary_file.read_text())
        key = summary_file.stem.replace("benchmark_", "").replace("_summary", "")
        bench["baselines"][key] = summary
    BENCH_SUMMARY_PATH.write_text(json.dumps(bench, indent=1, sort_keys=True) + "\n")
    log(f"benchmark baselines -> {BENCH_SUMMARY_PATH} ({sorted(bench['baselines'])})")

    baseline_report = {
        "artifact": "phase16_agent01_adversarial_baseline_v1",
        **PHASE16_STATUS_MARKERS,
        "written_utc": utc_now(),
        "pack": "phase16_adversarial_baseline_v1",
        "presets": {},
    }
    for pack_file in sorted(PACK_DIR.glob("adversarial_*.jsonl")):
        preset = pack_file.stem.replace("adversarial_", "")
        rows = _read_rows(pack_file)
        if rows:
            baseline_report["presets"][preset] = analyse_baseline(rows)
    BASELINE_SUMMARY_PATH.write_text(
        json.dumps(baseline_report, indent=1, sort_keys=True) + "\n"
    )
    log(
        f"adversarial baseline -> {BASELINE_SUMMARY_PATH} "
        f"({sorted(baseline_report['presets'])})"
    )
    return 0


# ---------------------------------------------------------------------------
# handoff
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def role_handoff(arguments) -> int:
    from stratego.evaluation.phase16 import adversarial, baseline, benchmark
    from stratego.evaluation.phase16.contract import (
        MEASUREMENT_HANDOFF_ARTIFACT,
        OPERATOR_LOG_SCHEMA,
        PHASE16_STATUS_MARKERS,
        QUICK_SUBSET_NAME,
        RUNNER_VERSION,
    )

    root = REPOSITORY_ROOT
    findings = []

    bench_manifest = benchmark.load_benchmark_manifest(root=root)
    library = adversarial.load_library(root=root)
    base_manifest = baseline.load_baseline_manifest(root=root)
    if base_manifest["adversarial_library_digest"] != library["library_digest"]:
        findings.append(
            "baseline manifest binds a different adversarial library digest"
        )

    baselines = {}
    if BENCH_SUMMARY_PATH.is_file():
        summary = json.loads(BENCH_SUMMARY_PATH.read_text())
        for key, entry in summary.get("baselines", {}).items():
            baselines[key] = {
                "pack": entry["pack"],
                "subset": entry["subset"],
                "preset": entry["preset"],
                "games": entry["games"],
                "ewr": entry["summary"]["ewr"],
                "wins": entry["summary"]["wins"],
                "draws": entry["summary"]["draws"],
                "losses": entry["summary"]["losses"],
            }
    adversarial_readings = {}
    if BASELINE_SUMMARY_PATH.is_file():
        summary = json.loads(BASELINE_SUMMARY_PATH.read_text())
        for preset, entry in summary.get("presets", {}).items():
            primary = entry.get("paired", {}).get(
                "adversarial_opponent_minus_control", {}
            )
            adversarial_readings[preset] = {
                "pack": "phase16_adversarial_baseline_v1",
                "games": entry["games"],
                "arm_ewr": {
                    arm: block["ewr"] for arm, block in entry["arms"].items()
                },
                "arm2_minus_arm1": primary.get("overall"),
                "drop": primary.get("drop"),
                "reading": entry.get("reading"),
            }

    handoff = {
        "artifact": MEASUREMENT_HANDOFF_ARTIFACT,
        **PHASE16_STATUS_MARKERS,
        "written_utc": utc_now(),
        "benchmark": {
            "artifact": bench_manifest["artifact"],
            "path": "data/phase16/phase16_benchmark_v1.json",
            "file_sha256": sha256_file(root / benchmark.DEFAULT_MANIFEST_PATH),
            "manifest_digest": bench_manifest["manifest_digest"],
            "board_count": bench_manifest["board_count"],
            "quick_subset": QUICK_SUBSET_NAME,
        },
        "adversarial_library": {
            "artifact": library["artifact"],
            "path": "data/phase16/phase16_adversarial_setups_v1.json",
            "file_sha256": sha256_file(root / adversarial.DEFAULT_LIBRARY_PATH),
            "library_digest": library["library_digest"],
            "authored_digest": library["authored_digest"],
            "setup_count": library["setup_count"],
            "harvest_revision": library["harvest_revision"],
            "operator_harvest_count": library["families"]["operator_harvest"][
                "setup_count"
            ],
            "note": (
                "operator_harvest appends bump harvest_revision and library_digest; "
                "authored_digest is frozen"
            ),
        },
        "adversarial_baseline_pack": {
            "artifact": base_manifest["artifact"],
            "path": "data/phase16/phase16_adversarial_baseline_v1.json",
            "file_sha256": sha256_file(root / baseline.DEFAULT_MANIFEST_PATH),
            "manifest_digest": base_manifest["manifest_digest"],
            "board_count": base_manifest["board_count"],
        },
        "scoring_runner": {
            "version": RUNNER_VERSION,
            "entry_point": (
                "stratego.evaluation.phase16.runner.score_on_benchmark"
                "(mode_or_provider, preset, workers, subset=None)"
            ),
            "provider_interface": (
                "a Phase 15 production pairing id, or {'factory': "
                "'module:callable', 'kwargs': {...}, 'arm_id': ...}; the factory "
                "returns a seat with arm_id, pairing and decide(state, legal, "
                "spec, plan); the oracle is refused by name"
            ),
        },
        "baseline_numbers": {
            "benchmark": baselines,
            "adversarial": adversarial_readings,
            "note": "every EWR names its pack; cross-pack comparisons are forbidden",
        },
        "operator_log_schema": OPERATOR_LOG_SCHEMA,
        "operator_log_path": "data/phase16/operator_games.jsonl",
        "findings": findings,
        "verified": not findings,
    }
    HANDOFF_PATH.write_text(json.dumps(handoff, indent=1, sort_keys=True) + "\n")
    log(f"handoff -> {HANDOFF_PATH} (verified={handoff['verified']})")
    return 0 if not findings else 1


# ---------------------------------------------------------------------------
# backup
# ---------------------------------------------------------------------------


def role_backup(arguments) -> int:
    date_tag = datetime.datetime.now().strftime("%Y%m%d")
    archive = Path(f"/Volumes/Brandon_Washington/stratego_untracked_backup_{date_tag}.tar")
    if archive.exists():
        archive = Path(
            f"/Volumes/Brandon_Washington/stratego_untracked_backup_{date_tag}"
            f"_{datetime.datetime.now().strftime('%H%M')}.tar"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    untracked = [
        line[3:].strip() for line in status.stdout.splitlines() if line.startswith("??")
    ]
    if not untracked:
        raise SystemExit("no untracked paths — refusing to write an empty backup")
    log(f"archiving {len(untracked)} untracked roots to {archive}…")
    command = ["tar", "-cf", str(archive), "--exclude=*_prefix_*.npy", *untracked]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    listing = subprocess.run(
        ["tar", "-tf", str(archive)], capture_output=True, text=True, check=True
    )
    entries = [line for line in listing.stdout.splitlines() if line.strip()]
    filelist = Path(str(archive)[: -len(".tar")] + ".filelist.txt")
    filelist.write_text("\n".join(entries) + "\n")
    digest = sha256_file(archive)
    Path(str(archive)[: -len(".tar")] + ".sha256").write_text(
        f"{digest}  {archive}\n"
    )
    record = json.loads(BACKUP_RECORD_PATH.read_text())
    record["end_of_task_backup"] = {
        "performed_utc": utc_now(),
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "tar_entries": len(entries),
        "sha256": digest,
        "filelist_path": str(filelist),
        "excluded": ["*_prefix_*.npy (regenerable prefix caches)"],
        "untracked_roots": len(untracked),
        "status": "complete",
    }
    BACKUP_RECORD_PATH.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    log(
        f"backup complete: {archive} ({archive.stat().st_size / 1e9:.2f} GB, "
        f"{len(entries)} entries, sha256 {digest[:16]}…)"
    )
    return 0


# ---------------------------------------------------------------------------
# harvest
# ---------------------------------------------------------------------------


def role_harvest(arguments) -> int:
    from stratego.evaluation.phase16.operator_log import harvest_operator_setups

    report = harvest_operator_setups(root=REPOSITORY_ROOT)
    log(
        f"harvest: scanned {report['games_scanned']} games, found "
        f"{report['operator_setups_found']} operator setups, appended "
        f"{len(report['appended'])} new ({report['appended']}); "
        f"harvest revision {report['harvest_revision']}"
    )
    return 0


# ---------------------------------------------------------------------------


ROLES = {
    "build": role_build,
    "bench": role_bench,
    "baseline": role_baseline,
    "analyse": role_analyse,
    "handoff": role_handoff,
    "harvest": role_harvest,
    "backup": role_backup,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--arm", default="p24_b24", help="bench: the system to score")
    parser.add_argument("--preset", default="TINY")
    parser.add_argument("--subset", default=None, help="bench: quick60 for the quick subset")
    parser.add_argument(
        "--arms",
        default="benchmark_control,adversarial_opponent,adversarial_both",
        help="baseline: comma-separated arm names",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-hours", type=float, default=2.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    return ROLES[arguments.role](arguments)


if __name__ == "__main__":
    raise SystemExit(main())
