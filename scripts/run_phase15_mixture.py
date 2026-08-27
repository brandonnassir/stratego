#!/usr/bin/env python
"""Phase 15 belief-mixture pilot: gate, Stage 1, Stage 2, report.

    python scripts/run_phase15_mixture.py --role gate
    python scripts/run_phase15_mixture.py --role stage1 --workers 8
    python scripts/run_phase15_mixture.py --role stage2 --workers 10
    python scripts/run_phase15_mixture.py --role report

Nothing here trains, and nothing here changes the search. The pilot builds
one extra belief provider — `lambda * B24 + (1 - lambda) * remaining_count`,
normalized, through the accepted legal-world sampler — and asks whether it
rescues the deeper rung the previous pilot found to be a regression.

Stage 2 refuses to run unless Stage 1 selected an interior lambda under a
rule that was written down before the numbers were seen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.search.phase15.contract import PHASE15_STATUS_MARKERS  # noqa: E402

REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase15"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "phase15"

POSITION_MANIFEST = REPORT_ROOT / "agent_02_position_manifest.json"
BUDGET_PATH = REPORT_ROOT / "agent_02_budget_profile.json"
CANDIDATE_PATH = CHECKPOINT_ROOT / "phase15_search_candidate_v1.json"
DEEP_GAMES_JSONL = REPORT_ROOT / "agent_02_deep_games.jsonl"

GATE_PATH = REPORT_ROOT / "agent_02_mixture_gate.json"
STAGE1_PATH = REPORT_ROOT / "agent_02_mixture_stage1.json"
STAGE1_CSV = REPORT_ROOT / "agent_02_mixture_stage1_decisions.csv"
STAGE2_PATH = REPORT_ROOT / "agent_02_mixture_stage2.json"
STAGE2_JSONL = REPORT_ROOT / "agent_02_mixture_stage2_games.jsonl"
STAGE2_CSV = REPORT_ROOT / "agent_02_mixture_stage2_games.csv"
REPORT_PATH = REPORT_ROOT / "agent_02_mixture_report.md"


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _read(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; run the earlier role first")
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: "list[dict]") -> Path:
    if not rows:
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _progress(label: str, quiet: bool):
    started = time.perf_counter()

    def report(done: int, total: int, _payload=None) -> None:
        if quiet or (done % 10 and done != total):
            return
        elapsed = time.perf_counter() - started
        rate = elapsed / max(done, 1)
        print(
            f"[{label}] {done}/{total}  {elapsed / 60:.1f} min elapsed, "
            f"~{rate * (total - done) / 60:.1f} min left",
            flush=True,
        )

    return report


def _positions(limit: "int | None" = None):
    from stratego.search.phase15.boards import Phase15MatchSetupSources
    from stratego.search.phase15.positions import materialize_positions

    manifest = _read(POSITION_MANIFEST)
    replayed = materialize_positions(
        manifest, sources=Phase15MatchSetupSources(), verify=True
    )
    if limit is not None and limit < len(replayed):
        step = max(1, len(replayed) // int(limit))
        replayed = replayed[::step][: int(limit)]
    return manifest, replayed


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


def role_gate(args) -> dict:
    """Identity, budgets, mixture algebra, determinism, world legality."""
    import torch

    from stratego.search.phase15.candidate import load_candidate
    from stratego.search.phase15.loaders import load_all
    from stratego.search.phase15.mixture import (
        MIXTURE_LAMBDAS,
        MIXTURE_VERSION,
        check_configuration_invariants,
        check_determinism,
        check_frozen_identity,
        check_mixture_algebra,
        check_worlds_legal,
    )

    torch.set_num_threads(int(args.threads))
    started = time.perf_counter()
    candidate = load_candidate(CANDIDATE_PATH)
    models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=False)
    manifest, states = _positions(int(args.gate_positions))
    print(f"mixture gate on {len(states)} replayed positions", flush=True)

    checks = {
        "frozen_identity": check_frozen_identity(models, candidate),
        "configuration_invariants": check_configuration_invariants(),
        "mixture_algebra": check_mixture_algebra(models, states),
        "determinism_and_legality": check_determinism(models, states),
        "worlds_legal": check_worlds_legal(models, states),
    }
    payload = {
        "artifact": MIXTURE_VERSION + "_gate",
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "lambdas": list(MIXTURE_LAMBDAS),
        "position_manifest_digest": manifest["manifest_digest"],
        "positions": len(states),
        "checks": checks,
        "seconds": round(time.perf_counter() - started, 1),
        "passed": all(entry.get("passed") for entry in checks.values()),
    }
    _write(GATE_PATH, payload)
    print(f"mixture gate: {'PASS' if payload['passed'] else 'FAIL'}", flush=True)
    for name, entry in checks.items():
        print(f"  {'ok  ' if entry.get('passed') else 'FAIL'} {name}", flush=True)
        for finding in entry.get("findings") or []:
            print(f"       {finding}", flush=True)
    if not payload["passed"]:
        raise SystemExit("the mixture gate failed; no stage may run")
    return payload


def _stage1_chunk(bounds):
    """One worker's slice of Stage 1. Pure function of its position range."""
    import torch

    from stratego.search.phase15.loaders import load_all
    from stratego.search.phase15.mixture_pilot import run_stage1

    start, stop, device, threads = bounds
    torch.set_num_threads(int(threads))
    models = load_all(root=REPOSITORY_ROOT, device=device, with_anchor=False)
    _manifest, states = _positions(None)
    return run_stage1(models, states[start:stop], device=device)


def role_stage1(args) -> dict:
    """The cheap position diagnostic: every arm, every position, one seed."""
    from concurrent.futures import ProcessPoolExecutor

    from stratego.search.phase15.mixture import (
        MIXTURE_DECISION_SEED,
        MIXTURE_LAMBDAS,
        MIXTURE_STAGE1_PRESET,
        MIXTURE_VERSION,
    )
    from stratego.search.phase15.mixture_pilot import (
        check_endpoint_identity,
        check_shared_candidate_set,
        select_lambda,
        summarize_stage1,
    )

    gate = _read(GATE_PATH)
    if not gate.get("passed"):
        raise SystemExit("the mixture gate has not passed")

    manifest, states = _positions(int(args.positions) if args.positions else None)
    total = len(states)
    workers = max(1, int(args.workers))
    print(f"stage 1: {total} positions x 9 arms over {workers} workers", flush=True)

    started = time.perf_counter()
    if workers == 1:
        rows = _stage1_chunk((0, total, args.device, args.threads))
    else:
        edges = [round(index * total / workers) for index in range(workers + 1)]
        chunks = [
            (edges[index], edges[index + 1], args.device, args.threads)
            for index in range(workers)
            if edges[index] < edges[index + 1]
        ]
        rows = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            for done, part in enumerate(pool.map(_stage1_chunk, chunks), start=1):
                rows.extend(part)
                print(
                    f"[stage 1] chunk {done}/{len(chunks)} done, "
                    f"{(time.perf_counter() - started) / 60:.1f} min elapsed",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["position_id"], row["arm"]))
    _write_csv(STAGE1_CSV, rows)

    summary = summarize_stage1(rows)
    endpoint = check_endpoint_identity(rows)
    shared = check_shared_candidate_set(rows)
    selection = select_lambda(rows, summary)

    payload = {
        "artifact": MIXTURE_VERSION + "_stage1",
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "question": (
            "can mixing B24 with the robust remaining-count belief prevent the "
            "degradation seen when search goes deeper?"
        ),
        "stage": "stage_1_position_diagnostic",
        "preset": MIXTURE_STAGE1_PRESET,
        "seed": MIXTURE_DECISION_SEED,
        "lambdas": list(MIXTURE_LAMBDAS),
        "positions": total,
        "position_manifest_digest": manifest["manifest_digest"],
        "decisions": len(rows),
        "arms": summary,
        "checks": {
            "endpoint_identity": endpoint,
            "shared_candidate_set": shared,
        },
        "selection": selection,
        "workers": workers,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    _write(STAGE1_PATH, payload)

    print("\nstage 1 summary (oracle at LARGE is the reference):", flush=True)
    print(
        f"  {'arm':18s} {'agree':>7s} {'regretQ':>8s} {'med':>7s} "
        f"{'!=MED':>7s} {'!=LRG':>7s} {'illegal':>7s} {'err':>4s} {'s/move':>7s}",
        flush=True,
    )
    for arm, entry in summary.items():
        print(
            f"  {arm:18s} {_fmt(entry['oracle_agreement']):>7s} "
            f"{_fmt(entry['oracle_q_regret_mean'], 4):>8s} "
            f"{_fmt(entry['oracle_q_regret_median'], 4):>7s} "
            f"{_fmt(entry['disagreement_with_b24_medium']):>7s} "
            f"{_fmt(entry['disagreement_with_b24_large']):>7s} "
            f"{entry['illegal_decisions']:>7d} {entry['search_errors']:>4d} "
            f"{entry['median_seconds']:>7.3f}",
            flush=True,
        )
    print(
        f"\n  endpoint identity (lambda=1 == frozen p24_b24 at LARGE): "
        f"{'PASS' if endpoint['passed'] else 'FAIL'} "
        f"({endpoint['differing_positions']}/{endpoint['positions']} differ)",
        flush=True,
    )
    print(
        f"  shared candidate set: {'PASS' if shared['passed'] else 'FAIL'} "
        f"({shared['outside']} of {shared['decisions']} outside)",
        flush=True,
    )
    print(f"\n  selection: {selection['selected_arm'] or 'NO USEFUL MIXTURE'}", flush=True)
    for finding in selection["findings"]:
        print(f"    {finding}", flush=True)
    return payload


def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def role_stage2(args) -> dict:
    """The tiny match confirmation. Refuses to run without a selected lambda."""
    from stratego.search.phase15.mixture import (
        MIXTURE_REFERENCE_PRESET,
        MIXTURE_STAGE1_PRESET,
        MIXTURE_VERSION,
        mixture_arm_id,
    )
    from stratego.search.phase15.mixture_pilot import MixTask, run_stage2_pack

    stage1 = _read(STAGE1_PATH)
    selection = stage1["selection"]
    if not selection.get("stage2_authorized") and not args.force:
        raise SystemExit(
            "stage 1 selected no interior mixture, so stage 2 does not run; "
            "the experiment closes with 'no useful mixture'"
        )
    lam = float(selection["selected_lambda"] if not args.force else args.force_lambda)
    mix_arm = mixture_arm_id(lam)

    boards = list(_read(BUDGET_PATH)["board_ids"])
    if args.boards and int(args.boards) < len(boards):
        step = max(1, len(boards) // int(args.boards))
        boards = boards[::step][: int(args.boards)]
    print(
        f"stage 2: lambda={lam:.2f}, {len(boards)} paired boards, three arms",
        flush=True,
    )

    tasks = (
        [
            MixTask("p24_b24", MIXTURE_REFERENCE_PRESET, board)
            for board in boards
        ]
        + [MixTask("p24_b24", MIXTURE_STAGE1_PRESET, board) for board in boards]
        + [
            MixTask(mix_arm, MIXTURE_STAGE1_PRESET, board, lam=lam)
            for board in boards
        ]
    )
    started = time.perf_counter()
    if STAGE2_JSONL.exists() and not args.append:
        STAGE2_JSONL.unlink()
    results = run_stage2_pack(
        tasks,
        root=str(REPOSITORY_ROOT),
        device=args.device,
        workers=int(args.workers),
        progress=_progress("stage 2", args.quiet),
    )
    STAGE2_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(STAGE2_JSONL, "a") as handle:
        for entry in results:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    payload = {
        "artifact": MIXTURE_VERSION + "_stage2_pack",
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "selected_lambda": lam,
        "mixture_arm": mix_arm,
        "arms": ["p24_b24|MEDIUM", "p24_b24|LARGE", f"{mix_arm}|LARGE"],
        "boards": len(boards),
        "board_ids": boards,
        "games_played": len(results),
        "workers": int(args.workers),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    _write(STAGE2_PATH, payload)
    print(
        f"stage 2: {len(results)} games in {payload['wall_seconds'] / 60:.1f} min",
        flush=True,
    )
    return payload


def role_report(args) -> dict:
    """Read whatever stages ran, apply the decision rule, and write it up."""
    from stratego.search.phase15.mixture_report_text import build_mixture_report

    from stratego.search.phase15.mixture_pilot import reference_comparisons

    gate = _read(GATE_PATH)
    stage1 = _read(STAGE1_PATH)

    # The decision CSV is the primary per-decision record; the reference
    # comparisons are derived from it rather than re-played, so adding them
    # costs nothing and cannot disagree with the rows they came from.
    if STAGE1_CSV.is_file() and "reference_comparisons" not in stage1:
        with open(STAGE1_CSV) as handle:
            rows = list(csv.DictReader(handle))
        stage1["reference_comparisons"] = reference_comparisons(rows)
        _write(STAGE1_PATH, stage1)

    stage2_pack = json.loads(STAGE2_PATH.read_text()) if STAGE2_PATH.is_file() else None

    stage2 = None
    if stage2_pack is not None and STAGE2_JSONL.is_file():
        from stratego.search.phase15.mixture_pilot import (
            analyse_stage2,
            check_reference_arms_reproduce,
            decide_stage2,
        )

        entries = [
            json.loads(line)
            for line in STAGE2_JSONL.read_text().splitlines()
            if line.strip()
        ]
        _write_csv(STAGE2_CSV, [entry["row"] for entry in entries])
        rungs = analyse_stage2(entries, "p24_b24|MEDIUM")
        stored = []
        if DEEP_GAMES_JSONL.is_file():
            stored = [
                json.loads(line)["row"]
                for line in DEEP_GAMES_JSONL.read_text().splitlines()
                if line.strip()
            ]
        stage2 = {
            "pack": stage2_pack,
            "rungs": rungs,
            "reference_arms_reproduce": check_reference_arms_reproduce(
                [entry["row"] for entry in entries], stored
            ),
            "decision": decide_stage2(
                rungs,
                medium_arm="p24_b24|MEDIUM",
                large_arm="p24_b24|LARGE",
                mix_arm=f"{stage2_pack['mixture_arm']}|LARGE",
            ),
        }
        _write(STAGE2_PATH, {**stage2_pack, "analysis": stage2})

    text = build_mixture_report(gate=gate, stage1=stage1, stage2=stage2)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    print(f"wrote {REPORT_PATH}", flush=True)
    return {"report": str(REPORT_PATH)}


ROLES = {
    "gate": role_gate,
    "stage1": role_stage1,
    "stage2": role_stage2,
    "report": role_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gate-positions", type=int, default=6)
    parser.add_argument("--positions", type=int, default=0)
    parser.add_argument("--boards", type=int, default=36)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-lambda", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    ROLES[args.role](args)
    print(
        f"[{args.role}] finished in {(time.perf_counter() - started) / 60:.2f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
