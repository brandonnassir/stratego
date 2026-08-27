#!/usr/bin/env python3
"""Phase 15 Agent 2 — P18/P24 belief-guided search integration.

Specification source:
`instructions/phase_15_belief_search_engineering/02_AGENT_2_SEARCH_IMPLEMENTATION.md`

This script controls nothing outside Phase 15. It sends no signal to any
process it did not start, creates no emergency-stop file, opens no Phase 14
run state or checkpoint, and invokes no closeout or finalization command. Its
only children are its own worker pool.

Roles, in the order they must run:

```text
boundary    inspect live process state, read-only
boards      build and write the fresh orientation-safe match manifest
positions   build and write the Stage A decision-position manifest
gate        the section 9 correctness gate; nothing may run before it passes
stage_a     the section 11 quick decision diagnostic
stage_b     the section 12 complete-system match comparison
stage_c     the section 13 budget ladder
select      the section 14 system matrix and selection
candidate   freeze phase15_search_candidate_v1 and load it back
report      write the Agent 2 report and summary
```
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.search.phase15.contract import (  # noqa: E402
    COMBINED_PAIRING_IDS,
    DIAGNOSTIC_PAIRING_IDS,
    LADDER_PRESET_NAMES,
    MATCH_LIBRARY_SPLIT,
    PHASE15_STATUS_MARKERS,
    PRODUCTION_PAIRING_IDS,
    pairing as pairing_of,
)

REPORT_ROOT = REPOSITORY_ROOT / "reports" / "phase15"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "phase15"

MATCH_MANIFEST = REPORT_ROOT / "agent_02_match_manifest.json"
POSITION_MANIFEST = REPORT_ROOT / "agent_02_position_manifest.json"
GATE_PATH = REPORT_ROOT / "agent_02_gate.json"
STAGE_A_PATH = REPORT_ROOT / "agent_02_stage_a.json"
STAGE_B_PATH = REPORT_ROOT / "agent_02_stage_b.json"
BUDGET_PATH = REPORT_ROOT / "agent_02_budget_profile.json"
MATRIX_PATH = REPORT_ROOT / "agent_02_system_matrix.json"
LATENCY_PATH = REPORT_ROOT / "agent_02_latency_pilot.json"
DEEP_GATE_PATH = REPORT_ROOT / "agent_02_deep_gate.json"
DEEP_PACK_PATH = REPORT_ROOT / "agent_02_deep_pack.json"
DEEP_GAMES_JSONL = REPORT_ROOT / "agent_02_deep_games.jsonl"
DEEP_GAMES_CSV = REPORT_ROOT / "agent_02_deep_games.csv"
DEEP_REPORT_PATH = REPORT_ROOT / "agent_02_deep_report.md"
DECISIONS_CSV = REPORT_ROOT / "agent_02_decisions.csv"
GAMES_JSONL = REPORT_ROOT / "agent_02_games.jsonl"
GAMES_CSV = REPORT_ROOT / "agent_02_games.csv"
BOUNDARY_PATH = REPORT_ROOT / "agent_02_process_boundary.json"
CANDIDATE_PATH = CHECKPOINT_ROOT / "phase15_search_candidate_v1.json"
REPORT_PATH = REPORT_ROOT / "agent_02_report.md"
SUMMARY_PATH = REPORT_ROOT / "agent_02_summary.json"


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _read(path: Path):
    if not path.is_file():
        raise SystemExit(f"missing prerequisite artifact: {path}")
    return json.loads(path.read_text())


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _environment() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def _progress(label: str, quiet: bool):
    started = time.perf_counter()

    def report(done: int, total: int, _payload=None) -> None:
        if quiet or (done % max(1, total // 40) and done != total):
            return
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        remaining = (total - done) / rate if rate else 0.0
        print(
            f"  {label}: {done}/{total} ({100 * done / total:5.1f}%) "
            f"{elapsed / 60:.1f}m elapsed, ~{remaining / 60:.1f}m left",
            flush=True,
        )

    return report


# ---------------------------------------------------------------------------
# 1. Process boundary
# ---------------------------------------------------------------------------


def role_boundary(args) -> dict:
    """Section 2: inspect live task state, read-only. No signals, ever."""
    processes = subprocess.run(
        ["ps", "-Ao", "pid,pcpu,pmem,etime,command"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    interesting = [
        line
        for line in processes
        if any(
            token in line
            for token in ("phase14", "phase15", "stratego", "caffeinate")
        )
        and "ps -Ao" not in line
        and "grep" not in line
    ]
    learner = [
        line
        for line in interesting
        if "phase14_launch" in line or "phase14_train" in line or "phase14_run" in line
    ]
    load = os.getloadavg()
    payload = {
        "artifact": "phase15_agent02_process_boundary_v1",
        **PHASE15_STATUS_MARKERS,
        "checked_utc": _utc(),
        "method": "read-only `ps` inspection; no signal, no control file, no run-state write",
        "matching_processes": interesting,
        "phase14_learner_or_evaluator_running": bool(learner),
        "phase14_processes": learner,
        "load_average": {"1m": load[0], "5m": load[1], "15m": load[2]},
        "cpu_count": os.cpu_count(),
        "planned_workers": int(args.workers),
        "verdict": "clear_to_run" if not learner else "would_compete_stop_for_review",
        "note": (
            "a read-only Phase 14 dashboard process is not a learner, evaluator or "
            "supervisor and does not compete for compute; it is listed but not "
            "treated as competition"
        ),
    }
    _write(BOUNDARY_PATH, payload)
    print(f"process boundary: {payload['verdict']}", flush=True)
    for line in interesting:
        print(f"  {line.strip()[:140]}", flush=True)
    if learner:
        raise SystemExit(
            "a Phase 14 learner/evaluator/supervisor is running; section 2 requires "
            "stopping for operator review rather than competing with it"
        )
    return payload


# ---------------------------------------------------------------------------
# 2. Boards and positions
# ---------------------------------------------------------------------------


def role_boards(args) -> dict:
    from stratego.search.phase15.boards import (
        Phase15MatchSetupSources,
        board_plans,
        build_manifest,
        materialize_manifest,
    )

    started = time.perf_counter()
    sources = Phase15MatchSetupSources()
    plans = board_plans(int(args.boards_per_cell), sources=sources)
    manifest = build_manifest(
        plans,
        generated_utc=_utc(),
        library_split=MATCH_LIBRARY_SPLIT,
        sources=sources,
        boards_per_cell=int(args.boards_per_cell),
        seconds=round(time.perf_counter() - started, 3),
    )
    # Reproducibility is a property of the pack, so prove it before storing it.
    materialize_manifest(manifest, sources=sources, verify=True)
    _write(MATCH_MANIFEST, manifest)
    print(
        f"match manifest: {manifest['board_count']} boards, digest "
        f"{manifest['manifest_digest'][:16]}",
        flush=True,
    )
    print(f"  balance: {json.dumps(manifest['balance']['by_setup_source'])}", flush=True)
    return manifest


def role_positions(args) -> dict:
    from stratego.search.phase15.boards import Phase15MatchSetupSources
    from stratego.search.phase15.loaders import load_all
    from stratego.search.phase15.matchplay import build_owners
    from stratego.search.phase15.positions import (
        build_manifest,
        generate_positions,
        materialize_positions,
    )

    started = time.perf_counter()
    models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=True)
    owners = build_owners(models, device=args.device)
    sources = Phase15MatchSetupSources()
    positions = generate_positions(
        owners,
        games_per_observer=int(args.position_games),
        sources=sources,
        progress=_progress("positions", args.quiet),
    )
    manifest = build_manifest(
        positions,
        generated_utc=_utc(),
        library_split=MATCH_LIBRARY_SPLIT,
        games_per_observer=int(args.position_games),
        seconds=round(time.perf_counter() - started, 3),
    )
    materialize_positions(manifest, sources=sources, verify=True)
    _write(POSITION_MANIFEST, manifest)
    print(
        f"position manifest: {manifest['position_count']} positions, digest "
        f"{manifest['manifest_digest'][:16]}",
        flush=True,
    )
    return manifest


# ---------------------------------------------------------------------------
# 3. The correctness gate
# ---------------------------------------------------------------------------


def role_gate(args) -> dict:
    from stratego.search.phase15.boards import Phase15MatchSetupSources
    from stratego.search.phase15.gate import run_gate
    from stratego.search.phase15.loaders import load_all
    from stratego.search.phase15.matchplay import build_owners

    models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=True)
    owners = build_owners(models, device=args.device)
    sources = Phase15MatchSetupSources()
    result = run_gate(
        models,
        owners,
        sources,
        games=int(args.gate_games),
        per_game=int(args.gate_positions),
    )
    summary = result.summary()
    summary["artifact"] = "phase15_agent02_correctness_gate_v1"
    summary.update(PHASE15_STATUS_MARKERS)
    summary["checked_utc"] = _utc()
    summary["identities"] = models.identities()

    regression = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/search", "-q"],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    tail = regression.stdout.strip().splitlines()[-1] if regression.stdout.strip() else ""
    summary["checks"]["phase12_frozen_candidate_regression"] = {
        "passed": regression.returncode == 0,
        "command": "pytest tests/search -q",
        "result": tail,
    }
    summary["passed"] = all(
        entry.get("passed") for entry in summary["checks"].values()
    )
    summary["checks_run"] = len(summary["checks"])
    summary["checks_passed"] = sum(
        1 for entry in summary["checks"].values() if entry.get("passed")
    )
    summary["failed"] = sorted(
        name for name, entry in summary["checks"].items() if not entry.get("passed")
    )
    _write(GATE_PATH, summary)
    print(
        f"correctness gate: {'PASS' if summary['passed'] else 'FAIL'} "
        f"({summary['checks_passed']}/{summary['checks_run']}) in {summary['seconds']}s",
        flush=True,
    )
    for name, entry in summary["checks"].items():
        mark = "ok " if entry.get("passed") else "FAIL"
        print(f"  {mark} {name}", flush=True)
        for finding in entry.get("findings") or []:
            print(f"       {finding}", flush=True)
    if not summary["passed"]:
        raise SystemExit("the section 9 correctness gate failed; no pack may run")
    return summary


# ---------------------------------------------------------------------------
# 4. Stage A
# ---------------------------------------------------------------------------


def role_stage_a(args) -> dict:
    from stratego.search.phase15.boards import Phase15MatchSetupSources
    from stratego.search.phase15.decisions import (
        DECISION_SEED,
        DECISION_VERSION,
        interpret,
        run_decisions,
        summarize,
    )
    from stratego.search.phase15.loaders import load_all
    from stratego.search.phase15.positions import materialize_positions

    gate = _read(GATE_PATH)
    if not gate.get("passed"):
        raise SystemExit("the correctness gate has not passed; Stage A may not run")
    manifest = _read(POSITION_MANIFEST)
    sources = Phase15MatchSetupSources()
    replayed = materialize_positions(manifest, sources=sources, verify=True)
    models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=False)

    started = time.perf_counter()
    rows = run_decisions(
        models,
        replayed,
        preset=args.stage_a_preset,
        progress=_progress("stage A", args.quiet),
    )
    summary = summarize(rows)
    payload = {
        "artifact": DECISION_VERSION,
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "preset": args.stage_a_preset,
        "seed": DECISION_SEED,
        "position_manifest_digest": manifest["manifest_digest"],
        "positions": len(replayed),
        "arms": summary,
        "interpretation": interpret(summary),
        "seconds": round(time.perf_counter() - started, 2),
        "note": (
            "a decision diagnostic on replayed positions, not a strength claim. The "
            "oracle arm reads hidden truth and is never deployable."
        ),
    }
    DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].row()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.row())
    _write(STAGE_A_PATH, payload)
    print(f"stage A: {len(rows)} decisions on {len(replayed)} positions", flush=True)
    for arm_id, entry in sorted(summary.items()):
        print(
            f"  {arm_id:24s} change={entry['move_change_rate_vs_direct']:.3f} "
            f"oracle_agree={entry['oracle_agreement']} "
            f"median={entry['median_seconds']:.3f}s",
            flush=True,
        )
    for move_model, reading in payload["interpretation"].items():
        print(f"  {move_model}: {reading['reading']}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# 5. Stage B
# ---------------------------------------------------------------------------


def _append_games(results: "list[dict]") -> None:
    GAMES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(GAMES_JSONL, "a") as handle:
        for entry in results:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _rewrite_games_csv() -> int:
    rows = []
    with open(GAMES_JSONL) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line)["row"])
    if not rows:
        return 0
    fieldnames = sorted({key for row in rows for key in row})
    with open(GAMES_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _load_games() -> "list[dict]":
    if not GAMES_JSONL.is_file():
        raise SystemExit(f"missing prerequisite artifact: {GAMES_JSONL}")
    entries = []
    with open(GAMES_JSONL) as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def role_stage_b(args) -> dict:
    from stratego.search.phase15.analysis import ANALYSIS_VERSION, analyse_pack
    from stratego.search.phase15.execution import Task, run_pack

    gate = _read(GATE_PATH)
    if not gate.get("passed"):
        raise SystemExit("the correctness gate has not passed; Stage B may not run")
    manifest = _read(MATCH_MANIFEST)
    board_ids = [row["board_id"] for row in manifest["boards"]]

    arms = list(PRODUCTION_PAIRING_IDS)
    if not args.no_oracle:
        arms += list(DIAGNOSTIC_PAIRING_IDS)
    # Every task carries the Stage B preset name. A direct arm ignores it (its
    # bundle has no config at all) and its stored row is relabelled `direct`
    # after the run, so the ladder can never mistake it for a search rung.
    tasks = [
        Task(
            arm_id=arm_id,
            preset_name=args.stage_b_preset,
            board_id=board,
            probe=(index % max(1, args.probe_every) == 0),
        )
        for arm_id in arms
        for index, board in enumerate(board_ids)
    ]

    started = time.perf_counter()
    if GAMES_JSONL.exists() and not args.append:
        GAMES_JSONL.unlink()
    results = run_pack(
        tasks,
        root=str(REPOSITORY_ROOT),
        device=args.device,
        workers=int(args.workers),
        progress=_progress("stage B", args.quiet),
    )
    for entry, task in zip(results, tasks):
        if pairing_of(task.arm_id).kind == "direct":
            entry["row"]["preset_id"] = "direct"
        entry["row"]["stage"] = "B"
    _append_games(results)
    total_rows = _rewrite_games_csv()

    summaries = analyse_pack(_load_games())
    probes = {}
    for entry, task in zip(results, tasks):
        if entry.get("probe") is None:
            continue
        bucket = probes.setdefault(
            task.arm_id,
            {
                "permutation_checks": 0,
                "permutation_assignments_changed": 0,
                "permutation_sensitive": 0,
                "direct_agreement_checks": 0,
                "failures": [],
                "expects_hidden_truth": entry["probe"]["expects_hidden_truth"],
            },
        )
        for key in (
            "permutation_checks",
            "permutation_assignments_changed",
            "permutation_sensitive",
            "direct_agreement_checks",
        ):
            bucket[key] += entry["probe"][key]
        bucket["failures"].extend(entry["probe"]["failures"])
    for bucket in probes.values():
        bucket["passed"] = not bucket["failures"]

    payload = {
        "artifact": ANALYSIS_VERSION,
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "preset": args.stage_b_preset,
        "match_manifest_digest": manifest["manifest_digest"],
        "boards": len(board_ids),
        "arms": arms,
        "games_played": len(results),
        "games_recorded": total_rows,
        "workers": int(args.workers),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "summaries": summaries,
        "probes": probes,
        "probe_passed": all(bucket["passed"] for bucket in probes.values()),
        "note": (
            "a compact paired engineering pack. No significance claim is made; each "
            "slice carries its own game count and the paired deltas carry a standard "
            "error so a reader can see what this sample can and cannot resolve."
        ),
    }
    _write(STAGE_B_PATH, payload)
    print(
        f"stage B: {len(results)} games over {len(arms)} arms in "
        f"{payload['wall_seconds'] / 60:.1f} min",
        flush=True,
    )
    for key in sorted(summaries):
        entry = summaries[key]
        paired = (entry.get("paired_vs_direct") or {}).get("delta")
        print(
            f"  {key:34s} EWR={entry['ewr']:.4f} "
            f"paired={paired if paired is not None else '   -  '} "
            f"worst_opp={(entry['min_opponent'] or {}).get('ewr')} "
            f"med={entry.get('median_seconds_per_move')}",
            flush=True,
        )
    return payload


# ---------------------------------------------------------------------------
# 6. Stage C
# ---------------------------------------------------------------------------


def role_stage_c(args) -> dict:
    from stratego.search.phase15.analysis import analyse_pack, paired_delta
    from stratego.search.phase15.budget import (
        BUDGET_VERSION,
        ladder_analysis,
        ladder_points,
        maximum_strength_mode,
        select_budget,
        strong_gate,
    )
    from stratego.search.phase15.execution import Task, run_pack

    stage_b = _read(STAGE_B_PATH)
    manifest = _read(MATCH_MANIFEST)
    summaries = stage_b["summaries"]

    ranked = sorted(
        (
            (key, entry)
            for key, entry in summaries.items()
            if key.split("|")[0] in COMBINED_PAIRING_IDS
        ),
        key=lambda item: item[1]["ewr"],
        reverse=True,
    )
    chosen = [key.split("|")[0] for key, _ in ranked[: int(args.ladder_pairings)]]
    if args.ladder_pairing:
        chosen = list(args.ladder_pairing)
    print(f"stage C ladder pairings: {chosen}", flush=True)

    # The ladder runs on a fixed subset of the Stage B boards: the same boards
    # and the same per-decision seeds at every rung, which is what section 13
    # asks for, at a cost the expensive rungs can afford.
    #
    # The subset is taken *per cell*, never as a stride over the board list.
    # The list is opponent-major, so a stride that stops early would silently
    # drop whole opponents off the end of the ladder — exactly the strata
    # section 14 cares most about.
    from collections import defaultdict

    by_cell = defaultdict(list)
    for row in manifest["boards"]:
        by_cell[row["cell_index"]].append(row["board_id"])
    per_cell = max(1, int(args.ladder_boards) // max(1, len(by_cell)))
    boards = [
        board
        for cell in sorted(by_cell)
        for board in by_cell[cell][:per_cell]
    ]
    print(
        f"stage C boards: {len(boards)} ({per_cell} per cell over {len(by_cell)} cells)",
        flush=True,
    )

    tasks = []
    for pairing_id in chosen:
        for preset in LADDER_PRESET_NAMES:
            if preset == stage_b["preset"]:
                continue  # already measured in Stage B on a superset of these boards
            for board in boards:
                tasks.append(Task(arm_id=pairing_id, preset_name=preset, board_id=board))

    started = time.perf_counter()
    results = run_pack(
        tasks,
        root=str(REPOSITORY_ROOT),
        device=args.device,
        workers=int(args.workers),
        progress=_progress("stage C", args.quiet),
    )
    for entry in results:
        entry["row"]["stage"] = "C"
    _append_games(results)
    _rewrite_games_csv()

    # Restrict every rung — including the Stage B one — to the ladder boards,
    # so the three rungs are compared on exactly one board list.
    board_set = set(boards)
    ladder_games = [
        entry
        for entry in _load_games()
        if entry["row"]["board_id"] in board_set
        and (
            entry["row"]["arm_id"] in chosen
            or entry["row"]["arm_id"].endswith("_direct")
        )
    ]
    ladder_summaries = analyse_pack(ladder_games)

    profiles = {}
    for pairing_id in chosen:
        points = ladder_points(ladder_summaries, pairing_id)
        cheapest = LADDER_PRESET_NAMES[0]
        paired = {}
        for preset in points:
            left = [
                entry["row"]
                for entry in ladder_games
                if entry["row"]["arm_id"] == pairing_id
                and entry["row"]["preset_id"] == preset
            ]
            right = [
                entry["row"]
                for entry in ladder_games
                if entry["row"]["arm_id"] == pairing_id
                and entry["row"]["preset_id"] == cheapest
            ]
            paired[preset] = paired_delta(left, right)
        selection = select_budget(points)
        profiles[pairing_id] = {
            "ladder": ladder_analysis(points, paired),
            "selection": selection,
            "maximum_strength": maximum_strength_mode(
                points, selection["selected_preset"]
            ),
            "strong_gate": strong_gate(points),
        }

    payload = {
        "artifact": BUDGET_VERSION,
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "pairings": chosen,
        "ladder_boards": len(boards),
        "board_ids": boards,
        "presets": list(LADDER_PRESET_NAMES),
        "games_played": len(results),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "profiles": profiles,
        "note": (
            "one variable at a time: the same pairing, the same boards, the same "
            "per-decision seeds, three accepted presets. Not a grid search."
        ),
    }
    _write(BUDGET_PATH, payload)
    print(f"stage C: {len(results)} games in {payload['wall_seconds'] / 60:.1f} min", flush=True)
    for pairing_id, profile in profiles.items():
        print(f"  {pairing_id}:", flush=True)
        for rung in profile["ladder"]["rungs"]:
            print(
                f"    {rung['preset_id']:7s} EWR={rung['ewr']:.4f} "
                f"med={rung['median_seconds_per_move']}s "
                f"gain/s={rung['ewr_gain_per_added_search_second']} "
                f"{rung['human_play']['verdict']}",
                flush=True,
            )
        print(
            f"    -> selected {profile['selection']['selected_preset']}, max "
            f"{profile['maximum_strength']['mode']}, STRONG allowed="
            f"{profile['strong_gate']['allowed']}",
            flush=True,
        )
    return payload


# ---------------------------------------------------------------------------
# 7. Selection
# ---------------------------------------------------------------------------


def role_select(args) -> dict:
    from stratego.search.phase15.analysis import select_system, system_matrix

    stage_b = _read(STAGE_B_PATH)
    budget = _read(BUDGET_PATH)
    matrix = system_matrix(stage_b["summaries"], preset_id=stage_b["preset"])
    selection = select_system(matrix)

    selected_pairing = selection["selected"]
    profile = budget["profiles"].get(selected_pairing)
    if profile is None:
        # The ladder was run on the Stage B leaders; if the section 14 decision
        # order lands elsewhere, say so plainly and use the Stage B preset.
        selected_preset = stage_b["preset"]
        maximum_preset = stage_b["preset"]
        budget_note = (
            f"{selected_pairing} was not on the Stage C ladder, so its budget is the "
            f"Stage B preset {stage_b['preset']}"
        )
    else:
        selected_preset = profile["selection"]["selected_preset"]
        maximum_preset = profile["maximum_strength"]["mode"]
        budget_note = profile["selection"]["rule"]

    payload = {
        "artifact": "phase15_system_matrix_v1",
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "preset": stage_b["preset"],
        "match_manifest_digest": stage_b["match_manifest_digest"],
        "matrix": matrix,
        "selection": selection,
        "selected_pairing": selected_pairing,
        "selected_preset": selected_preset,
        "maximum_strength_preset": maximum_preset,
        "budget_note": budget_note,
        "decision_order": [
            "reject any system with an information leak, illegal worlds/actions, "
            "identity mismatch or unstable fallback (the section 9 gate)",
            "prefer better overall and worst-stratum match strength on the fresh pack",
            "weight aggressive, unusual, Scout, Miner/Bomb and Flag-structure play",
            "on an effective tie prefer lower latency and the simpler belief pairing",
            "retain a maximum-strength mode when the slower one buys an observed gain",
        ],
    }
    _write(MATRIX_PATH, payload)
    print(f"selected: {selected_pairing} at {selected_preset} (max {maximum_preset})", flush=True)
    def _num(value, spec=".4f"):
        return "-" if value is None else format(value, spec)

    for pairing_id, entry in matrix.items():
        print(
            f"  {pairing_id:10s} direct={_num(entry['direct_ewr'])} "
            f"search={_num(entry['search_ewr'])} "
            f"paired={_num(entry['paired_delta_vs_direct'], '+.4f')} "
            f"worst={_num((entry['worst_opponent'] or {}).get('ewr'), '.4f')} "
            f"p95={_num(entry['p95_seconds_per_move'], '.4f')}",
            flush=True,
        )
    return payload


# ---------------------------------------------------------------------------
# 7b. The un-contended latency pilot
# ---------------------------------------------------------------------------


def role_latency(args) -> dict:
    """Measure per-move latency the way a human would actually experience it.

    The Stage B and Stage C numbers are measured with ten worker processes on
    fourteen cores, so every move time in them carries scheduler contention —
    they are the honest cost of the *pack*, not of a game. A person playing
    one game has the machine to themselves. This role therefore re-measures
    the selected pairing single-process, on replayed diagnostic positions, and
    it is that measurement the time caps are set from. Both numbers are kept:
    the contended one is what the pack cost, the un-contended one is what a
    move costs.
    """
    import numpy as np

    from stratego.search.phase15.boards import Phase15MatchSetupSources
    from stratego.search.phase15.contract import LADDER_PRESET_NAMES
    from stratego.search.phase15.loaders import load_all
    from stratego.search.phase15.positions import materialize_positions
    from stratego.search.phase15.systems import build_engine

    import torch

    torch.set_num_threads(int(args.latency_threads))

    matrix = _read(MATRIX_PATH)
    manifest = _read(POSITION_MANIFEST)
    pairing_ids = args.ladder_pairing or [matrix["selected_pairing"]]
    if matrix["selected_pairing"] not in pairing_ids:
        pairing_ids = [matrix["selected_pairing"], *pairing_ids]

    sources = Phase15MatchSetupSources()
    replayed = materialize_positions(manifest, sources=sources, verify=False)
    step = max(1, len(replayed) // int(args.latency_positions))
    sample = replayed[::step][: int(args.latency_positions)]

    models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=False)
    profiles = {}
    for pairing_id in pairing_ids:
        for preset in LADDER_PRESET_NAMES:
            bundle = build_engine(pairing_id, models, preset, device=args.device)
            timings = []
            forwards = []
            for _row, state, _plan in sample:
                started = time.perf_counter()
                decision = bundle.engine.choose_action(state, seed=20260824)
                timings.append(time.perf_counter() - started)
                forwards.append(decision.c1_forwards)
            array = np.asarray(timings, dtype=np.float64)
            profiles[f"{pairing_id}|{preset}"] = {
                "pairing_id": pairing_id,
                "preset_id": preset,
                "decisions": len(timings),
                "mean_seconds_per_move": round(float(array.mean()), 5),
                "median_seconds_per_move": round(float(np.median(array)), 5),
                "p95_seconds_per_move": round(float(np.percentile(array, 95)), 5),
                "p99_seconds_per_move": round(float(np.percentile(array, 99)), 5),
                "max_seconds_per_move": round(float(array.max()), 5),
                "mean_c1_forwards": round(float(np.mean(forwards)), 1),
            }
            print(
                f"  {pairing_id}|{preset:7s} median={profiles[f'{pairing_id}|{preset}']['median_seconds_per_move']:.3f}s "
                f"p95={profiles[f'{pairing_id}|{preset}']['p95_seconds_per_move']:.3f}s "
                f"max={profiles[f'{pairing_id}|{preset}']['max_seconds_per_move']:.3f}s "
                f"forwards={profiles[f'{pairing_id}|{preset}']['mean_c1_forwards']:.0f}",
                flush=True,
            )

    payload = {
        "artifact": "phase15_latency_pilot_v1",
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "measurement": (
            "one process, %d torch thread(s), no competing pack; the latency a "
            "single human-play or machine-vs-machine game actually sees"
        )
        % int(args.latency_threads),
        "positions": len(sample),
        "position_manifest_digest": manifest["manifest_digest"],
        "pairings": pairing_ids,
        "profiles": profiles,
        "note": (
            "the Stage B and Stage C move times are measured under ten-way "
            "process contention and are systematically slower; the caps are set "
            "from this pilot, and both numbers are reported"
        ),
    }
    _write(LATENCY_PATH, payload)
    return payload


# ---------------------------------------------------------------------------
# 8. The frozen candidate
# ---------------------------------------------------------------------------


def role_candidate(args) -> dict:
    from stratego.search.phase15.candidate import (
        build_candidate_record,
        load_player_from_candidate,
        write_candidate,
    )
    from stratego.search.phase15.loaders import load_all

    matrix = _read(MATRIX_PATH)
    stage_a = _read(STAGE_A_PATH)
    stage_b = _read(STAGE_B_PATH)
    budget = _read(BUDGET_PATH)
    gate = _read(GATE_PATH)
    positions = _read(POSITION_MANIFEST)
    pilot = _read(LATENCY_PATH)

    selected_pairing = matrix["selected_pairing"]
    selected_preset = matrix["selected_preset"]
    maximum_preset = matrix["maximum_strength_preset"]
    models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=False)

    def latency_of(preset_name: str) -> dict:
        entry = stage_b["summaries"].get(f"{selected_pairing}|{preset_name}")
        if entry is None:
            for profile in budget["profiles"].values():
                for rung in profile["ladder"]["rungs"]:
                    if rung["preset_id"] == preset_name:
                        return {
                            "median_seconds_per_move": rung["median_seconds_per_move"],
                            "p95_seconds_per_move": rung["p95_seconds_per_move"],
                            "max_seconds_per_move": rung["max_seconds_per_move"],
                            "search_seconds_per_game": rung["search_seconds_per_game"],
                        }
            return {}
        return {
            "median_seconds_per_move": entry.get("median_seconds_per_move"),
            "p95_seconds_per_move": entry.get("p95_seconds_per_move"),
            "max_seconds_per_move": entry.get("max_seconds_per_move"),
            "search_seconds_per_game": entry.get("search_seconds_per_game"),
            "move_latency": entry.get("move_latency"),
        }

    profile = budget["profiles"].get(selected_pairing) or {}
    ladder_latency = {}
    for rung in (profile.get("ladder") or {}).get("rungs", []):
        ladder_latency[rung["preset_id"]] = {
            "median_seconds_per_move": rung["median_seconds_per_move"],
            "p95_seconds_per_move": rung["p95_seconds_per_move"],
            "max_seconds_per_move": rung["max_seconds_per_move"],
        }

    def pilot_for(preset_name: str) -> dict:
        return pilot["profiles"].get(f"{selected_pairing}|{preset_name}", {})

    latency = {
        "under_pack_contention": {
            "selected_preset": latency_of(selected_preset)
            or ladder_latency.get(selected_preset, {}),
            "maximum_strength_preset": latency_of(maximum_preset)
            or ladder_latency.get(maximum_preset, {}),
            "measured_on": (
                "the fresh Phase 15 match pack, ten worker processes on fourteen "
                "cores; systematically slower than one game on an idle machine"
            ),
        },
        "single_process_pilot": {
            "selected_preset": pilot_for(selected_preset),
            "maximum_strength_preset": pilot_for(maximum_preset),
            "measured_on": pilot["measurement"],
            "positions": pilot["positions"],
        },
        "caps_derived_from": "single_process_pilot",
    }

    def cap_for(preset_name: str) -> float:
        # Headroom over the *un-contended* p95, the accepted Phase 12 shape:
        # enough to absorb scheduler jitter and thermal throttling without a
        # human opponent ever noticing a stall, and never above the 5 s
        # ceiling. Setting a cap from the ten-way-contended pack numbers would
        # buy headroom the deployed player does not need and would let a real
        # stall pass unnoticed.
        measured = pilot_for(preset_name)
        p95 = measured.get("p95_seconds_per_move")
        if p95 is None:
            raise SystemExit(
                f"the latency pilot measured no p95 for {selected_pairing}|"
                f"{preset_name}; run --role latency first"
            )
        return round(min(5.0, max(0.5, 3.5 * float(p95))), 2)

    time_caps = {
        "selected_search": cap_for(selected_preset),
        "maximum_strength": cap_for(maximum_preset),
    }

    limitations = [
        "a compact engineering pack: no significance claim is made and the "
        f"per-arm sample is {stage_b['boards']} paired boards",
        "the learned belief specialists did NOT consistently beat the "
        "remaining-count control: for P18 the count arm scored higher than both "
        "specialists, and for P24 only B24 beat it. The selected system is the "
        "strongest of the four P/B combinations, not evidence that a learned "
        "belief head is required; p18_remaining_count scored within 0.01 EWR of "
        "the selection while using no learned belief at all",
        "the oracle ceiling is small at this budget (+0.100 EWR for P18, +0.146 "
        "for P24), so most of the headroom in this search design is not in "
        "hidden-piece inference quality",
        "the match boards draw from the accepted setup library's `validation` "
        "split, which is also the population Agent 1's calibration and "
        "development corpora drew from; B18/B24 weights saw only the `train` "
        "split, so no belief model trained on these boards, but the two "
        "measurements are not independent draws of the base population",
        "the oracle arm is an offline ceiling diagnostic and is excluded from "
        "production by four independent refusals",
        "Stage A is a decision diagnostic on replayed positions, not a strength "
        "measurement",
        "no scientific validation phase was performed; this is an engineering "
        "selection",
    ]

    record = build_candidate_record(
        selected_pairing=selected_pairing,
        selected_preset=selected_preset,
        maximum_strength_preset=maximum_preset,
        models=models,
        time_caps=time_caps,
        latency=latency,
        match_manifest_digest=stage_b["match_manifest_digest"],
        position_manifest_digest=positions["manifest_digest"],
        gate={
            "gate_version": gate["gate_version"],
            "passed": gate["passed"],
            "checks_passed": gate["checks_passed"],
            "checks_run": gate["checks_run"],
        },
        stage_a={
            "positions": stage_a["positions"],
            "interpretation": stage_a["interpretation"],
            "arms": {
                key: {
                    "move_change_rate_vs_direct": entry["move_change_rate_vs_direct"],
                    "oracle_agreement": entry["oracle_agreement"],
                    "median_seconds": entry["median_seconds"],
                }
                for key, entry in stage_a["arms"].items()
            },
        },
        stage_b={
            "boards": stage_b["boards"],
            "games_played": stage_b["games_played"],
            "probe_passed": stage_b["probe_passed"],
            "selected_arm": stage_b["summaries"].get(
                f"{selected_pairing}|{stage_b['preset']}"
            ),
        },
        stage_c={
            "ladder_boards": budget["ladder_boards"],
            "pairings": budget["pairings"],
            "profile": profile,
        },
        system_matrix=matrix["matrix"],
        known_limitations=limitations,
        environment=_environment(),
        generated_utc=_utc(),
    )
    write_candidate(record, CANDIDATE_PATH)

    player, reloaded = load_player_from_candidate(
        CANDIDATE_PATH, root=REPOSITORY_ROOT, device=args.device
    )
    description = player.describe()
    print(f"frozen candidate: {CANDIDATE_PATH}", flush=True)
    print(
        f"  {reloaded['selected_system']['pairing_id']} at "
        f"{reloaded['search']['selected_preset']}, caps {time_caps}",
        flush=True,
    )
    print(f"  reload check: player modes {sorted(description['modes'])}", flush=True)
    return {"record": record, "player": description}


# ---------------------------------------------------------------------------
# 9. Report
# ---------------------------------------------------------------------------


def role_report(args) -> dict:
    from stratego.search.phase15.report_text import build_report, build_summary

    artifacts = {
        "boundary": _read(BOUNDARY_PATH),
        "match_manifest": _read(MATCH_MANIFEST),
        "position_manifest": _read(POSITION_MANIFEST),
        "gate": _read(GATE_PATH),
        "stage_a": _read(STAGE_A_PATH),
        "stage_b": _read(STAGE_B_PATH),
        "budget": _read(BUDGET_PATH),
        "matrix": _read(MATRIX_PATH),
        "latency_pilot": _read(LATENCY_PATH),
        "candidate": _read(CANDIDATE_PATH),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(artifacts))
    summary = build_summary(artifacts)
    _write(SUMMARY_PATH, summary)
    print(f"report: {REPORT_PATH}", flush=True)
    print(f"summary: {SUMMARY_PATH}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# 11. The deeper-search pilot
# ---------------------------------------------------------------------------


def _deep_states(args, models):
    """A sample of the diagnostic positions, replayed and verified."""
    from stratego.search.phase15.boards import Phase15MatchSetupSources
    from stratego.search.phase15.positions import materialize_positions

    manifest = _read(POSITION_MANIFEST)
    replayed = materialize_positions(
        manifest, sources=Phase15MatchSetupSources(), verify=True
    )
    step = max(1, len(replayed) // int(args.deep_positions))
    return manifest, replayed[::step][: int(args.deep_positions)]


def role_deep_gate(args) -> dict:
    """Identity, configuration control, determinism, legality and idle latency.

    Runs before any deeper game is played: the pilot's whole value depends on
    the stronger rungs being the same system with a bigger budget, and on the
    measurement being reproducible.
    """
    import torch

    from stratego.search.phase15.candidate import load_candidate
    from stratego.search.phase15.contract import DEEP_PILOT_PRESET_NAMES, DEEP_PILOT_VERSION
    from stratego.search.phase15.deep import (
        check_configuration_invariants,
        check_determinism,
        check_frozen_identity,
        check_worlds_legal,
        decision_divergence,
        latency_pilot,
    )
    from stratego.search.phase15.loaders import load_all

    torch.set_num_threads(int(args.latency_threads))
    started = time.perf_counter()
    candidate = load_candidate(CANDIDATE_PATH)
    models = load_all(root=REPOSITORY_ROOT, device=args.device, with_anchor=False)
    manifest, states = _deep_states(args, models)
    print(f"deep gate on {len(states)} replayed positions", flush=True)

    checks = {
        "frozen_identity": check_frozen_identity(models, candidate),
        "configuration_invariants": check_configuration_invariants(),
        "determinism_and_legality": check_determinism(models, states[: int(args.deep_gate_positions)]),
        "worlds_legal": check_worlds_legal(models, states[: int(args.deep_gate_positions)]),
    }
    latency = latency_pilot(models, states)
    divergence = decision_divergence(models, states)

    payload = {
        "artifact": DEEP_PILOT_VERSION + "_gate",
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "pairing_id": candidate["selected_system"]["pairing_id"],
        "presets": list(DEEP_PILOT_PRESET_NAMES),
        "position_manifest_digest": manifest["manifest_digest"],
        "positions": len(states),
        "checks": checks,
        "idle_latency": latency,
        "decision_divergence": divergence,
        "latency_threads": int(args.latency_threads),
        "seconds": round(time.perf_counter() - started, 1),
        "passed": all(entry.get("passed") for entry in checks.values()),
    }
    _write(DEEP_GATE_PATH, payload)

    print(f"deep gate: {'PASS' if payload['passed'] else 'FAIL'}", flush=True)
    for name, entry in checks.items():
        print(f"  {'ok  ' if entry.get('passed') else 'FAIL'} {name}", flush=True)
        for finding in entry.get("findings") or []:
            print(f"       {finding}", flush=True)
    print("  idle latency and measured compute:", flush=True)
    for name in DEEP_PILOT_PRESET_NAMES:
        entry = latency[name]
        print(
            f"    {name:7s} median={entry['median_seconds_per_move']:6.3f}s "
            f"p95={entry['p95_seconds_per_move']:6.3f}s "
            f"max={entry['max_seconds_per_move']:6.3f}s "
            f"fwd={entry['mean_c1_forwards']:8.1f} "
            f"measured={entry['measured_forward_ratio_vs_medium']:.2f}x "
            f"(naive {entry['naive_ratio_vs_medium']:.2f}x) "
            f"uniq={entry['mean_world_uniqueness']:.3f} "
            f"differs_from_MEDIUM={divergence[name]['fraction_differing_from_medium']:.3f}",
            flush=True,
        )
    if not payload["passed"]:
        raise SystemExit("the deeper-search gate failed; no deeper pack may run")
    return payload


def role_deep_pack(args) -> dict:
    """Play LARGE and XLARGE on the Stage C board list. MEDIUM is reused."""
    from stratego.search.phase15.contract import DEEP_PILOT_PAIRING, DEEP_PILOT_PRESET_NAMES
    from stratego.search.phase15.execution import Task, run_pack

    gate = _read(DEEP_GATE_PATH)
    if not gate.get("passed"):
        raise SystemExit("the deeper-search gate has not passed")
    budget = _read(BUDGET_PATH)
    boards = list(budget["board_ids"])
    print(
        f"deep pack: {len(boards)} boards (the Stage C list, one per cell), "
        f"pairing {DEEP_PILOT_PAIRING}",
        flush=True,
    )

    # All three rungs are played fresh in one pack, MEDIUM included. MEDIUM
    # already exists from Stage C on exactly these boards and seeds, and reusing
    # it would have been sound — but replaying it costs 14% of this pilot and
    # removes a whole class of doubt: every paired delta is then computed from
    # rows produced by identical code under identical conditions, and the fresh
    # MEDIUM rows double as a cross-run determinism proof against Stage C's.
    presets = list(DEEP_PILOT_PRESET_NAMES)
    arms = [DEEP_PILOT_PAIRING]
    if args.deep_oracle:
        # Essentially free: the oracle's worlds all collapse to one truth, so
        # its cost does not grow with the world budget at all.
        arms.append("p24_oracle")
    tasks = [
        Task(arm_id=arm, preset_name=name, board_id=board)
        for arm in arms
        for name in presets
        for board in boards
    ]
    started = time.perf_counter()
    if DEEP_GAMES_JSONL.exists() and not args.append:
        DEEP_GAMES_JSONL.unlink()
    results = run_pack(
        tasks,
        root=str(REPOSITORY_ROOT),
        device=args.device,
        workers=int(args.workers),
        progress=_progress("deep pack", args.quiet),
        keep_moves=True,
    )
    for entry in results:
        entry["row"]["stage"] = "deep"
    DEEP_GAMES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(DEEP_GAMES_JSONL, "a") as handle:
        for entry in results:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    payload = {
        "artifact": "phase15_deep_search_pack_v1",
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "pairing_id": DEEP_PILOT_PAIRING,
        "presets_played": presets,
        "arms": arms,
        "boards": len(boards),
        "board_ids": boards,
        "games_played": len(results),
        "workers": int(args.workers),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "medium_source": (
            "replayed fresh in this pack on the same boards, seeds and opponents "
            "Stage C used; the two runs are cross-checked for exact agreement"
        ),
    }
    _write(DEEP_PACK_PATH, payload)
    print(
        f"deep pack: {len(results)} games in {payload['wall_seconds'] / 60:.1f} min",
        flush=True,
    )
    return payload


def role_deep_report(args) -> dict:
    """Read the pilot, apply its decision rule, and write it up."""
    from stratego.search.phase15.contract import DEEP_PILOT_PRESET_NAMES, DEEP_PILOT_VERSION
    from stratego.search.phase15.deep import (
        analyse_rungs,
        check_medium_reproduces,
        decide,
        first_divergence,
    )
    from stratego.search.phase15.deep_report_text import build_deep_report

    gate = _read(DEEP_GATE_PATH)
    pack = _read(DEEP_PACK_PATH)
    entries = [
        json.loads(line)
        for line in DEEP_GAMES_JSONL.read_text().splitlines()
        if line.strip()
    ]
    pilot_entries = [
        entry for entry in entries if entry["row"]["arm_id"] == pack["pairing_id"]
    ]
    oracle_entries = [
        entry for entry in entries if entry["row"]["arm_id"].endswith("_oracle")
    ]

    rungs = analyse_rungs(
        pilot_entries, gate["idle_latency"], gate["decision_divergence"]
    )
    oracle_rungs = (
        analyse_rungs(oracle_entries, {}, {}) if oracle_entries else {}
    )

    rows_by_preset = {
        name: [entry["row"] for entry in pilot_entries if entry["row"]["preset_id"] == name]
        for name in DEEP_PILOT_PRESET_NAMES
    }
    divergence = first_divergence(rows_by_preset)

    stage_c_medium = [
        json.loads(line)["row"]
        for line in GAMES_JSONL.read_text().splitlines()
        if line.strip()
        and json.loads(line)["row"]["arm_id"] == pack["pairing_id"]
        and json.loads(line)["row"]["preset_id"] == "MEDIUM"
    ]
    reproduces = check_medium_reproduces(rows_by_preset["MEDIUM"], stage_c_medium)

    verdict = decide(rungs)
    payload = {
        "artifact": DEEP_PILOT_VERSION,
        **PHASE15_STATUS_MARKERS,
        "generated_utc": _utc(),
        "question": (
            "does buying roughly 2-4x more search compute make the selected "
            "P24 + B24 meaningfully stronger than MEDIUM, and what does it cost?"
        ),
        "pairing_id": pack["pairing_id"],
        "boards": pack["boards"],
        "games_played": pack["games_played"],
        "wall_seconds": pack["wall_seconds"],
        "gate_passed": gate["passed"],
        "gate_checks": {name: entry["passed"] for name, entry in gate["checks"].items()},
        "rungs": rungs,
        "oracle_reference": oracle_rungs,
        "first_divergence_from_medium": divergence,
        "medium_reproduces_stage_c": reproduces,
        "verdict": verdict.to_dict(),
        "ladder_closed": True,
        "note": (
            "a narrow paired pilot on one system. No architecture change, no "
            "training, no Phase 14 interaction, and the ladder is not extended "
            "beyond this."
        ),
    }
    _write(REPORT_ROOT / "agent_02_deep_pilot.json", payload)
    DEEP_REPORT_PATH.write_text(build_deep_report(payload, gate, pack))

    # The deep games, flattened for a spreadsheet like every other pack.
    rows = [entry["row"] for entry in entries]
    fieldnames = sorted({key for row in rows for key in row if key != "actions"})
    with open(DEEP_GAMES_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"deep pilot verdict: {verdict.recommendation} — {verdict.reason}", flush=True)
    for name in DEEP_PILOT_PRESET_NAMES:
        entry = rungs.get(name)
        if entry is None:
            continue
        paired = entry.get("paired_vs_medium") or {}
        idle = entry.get("idle_latency") or {}
        print(
            f"  {name:7s} EWR={entry['ewr']:.4f} "
            f"paired={('%+.4f' % paired['delta']) if paired.get('delta') is not None else '   -   '} "
            f"±{paired.get('standard_error', 0) or 0:.4f} "
            f"worst={(entry['worst_opponent'] or {}).get('ewr')} "
            f"idle_med={idle.get('median_seconds_per_move')} "
            f"idle_p95={idle.get('p95_seconds_per_move')} "
            f"differ={entry['moves_differing_from_medium'].get('fraction_differing_from_medium')} "
            f"fallbacks={entry['fallbacks']}",
            flush=True,
        )
    print(f"  MEDIUM reproduces Stage C: {reproduces['passed']} "
          f"({reproduces['boards_compared']} boards)", flush=True)
    return payload


# ---------------------------------------------------------------------------


ROLES = {
    "boundary": role_boundary,
    "boards": role_boards,
    "positions": role_positions,
    "gate": role_gate,
    "stage_a": role_stage_a,
    "stage_b": role_stage_b,
    "stage_c": role_stage_c,
    "select": role_select,
    "latency": role_latency,
    "candidate": role_candidate,
    "report": role_report,
    "deep_gate": role_deep_gate,
    "deep_pack": role_deep_pack,
    "deep_report": role_deep_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--boards-per-cell", type=int, default=2)
    parser.add_argument("--position-games", type=int, default=15)
    parser.add_argument("--gate-games", type=int, default=4)
    parser.add_argument("--gate-positions", type=int, default=3)
    parser.add_argument("--stage-a-preset", default="TINY")
    parser.add_argument("--stage-b-preset", default="TINY")
    parser.add_argument("--ladder-pairings", type=int, default=2)
    parser.add_argument("--ladder-pairing", action="append")
    parser.add_argument("--ladder-boards", type=int, default=60)
    parser.add_argument("--probe-every", type=int, default=10)
    parser.add_argument("--latency-positions", type=int, default=40)
    parser.add_argument("--latency-threads", type=int, default=1)
    parser.add_argument("--deep-positions", type=int, default=40)
    parser.add_argument("--deep-gate-positions", type=int, default=8)
    parser.add_argument("--deep-oracle", action="store_true")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    ROLES[args.role](args)
    print(f"[{args.role}] finished in {(time.perf_counter() - started) / 60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
