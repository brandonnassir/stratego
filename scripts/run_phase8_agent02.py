#!/usr/bin/env python3
"""Phase 8 Agent 2 acceptance harness: the deterministic synthetic corpus.

Verifies the Agent 1 prerequisite and every frozen upstream identity, generates
(or resumes) `synthetic_warmstart_corpus_v1`, finalizes it, audits the persisted
bytes independently of the generator, and writes the three Agent 2 artifacts:

    reports/phase_8_data/agent_02_corpus_manifest.json
    reports/phase_8_data/agent_02_corpus_audit.json
    reports/phase_8_data/agent_02_matchup_counts.csv

What this script is and is not
------------------------------
It produces the static corpus and the evidence that it is the corpus the
contract asked for. It builds no training example, trains nothing, and touches
no model: reconstruction and targets are Agent 3's deliverable and training is
Agents 4-6's.

Usage::

    python scripts/run_phase8_agent02.py --generate --finalize
    python scripts/run_phase8_agent02.py --generate --workers 10   # resumable
    python scripts/run_phase8_agent02.py --finalize                # audit only
    python scripts/run_phase8_agent02.py --finalize --run-pytest
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from stratego.training import rule_population as rp  # noqa: E402
from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training import warmstart_contract as wc  # noqa: E402
from stratego.training.corpus_commit import (  # noqa: E402
    CORPUS_COMMIT_VERSION,
    CorpusReader,
    corpus_content_digest,
    payload_digest,
)
from stratego.training.serialization import (  # noqa: E402
    DEFAULT_COMPRESSION_LEVEL,
    compress,
)
from stratego.training.trajectory import encode_game_record  # noqa: E402
from stratego.training.warmstart_seed import CORPUS_SPLITS  # noqa: E402

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"
MANIFEST_ARTIFACT = DATA_DIRECTORY / "agent_02_corpus_manifest.json"
AUDIT_ARTIFACT = DATA_DIRECTORY / "agent_02_corpus_audit.json"
MATCHUP_ARTIFACT = DATA_DIRECTORY / "agent_02_matchup_counts.csv"

#: The full pre-edit suite, measured before any Phase 8 Agent 2 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "3501 passed, 3 skipped in 200.81s (0:03:20)",
    "passed": 3501,
    "skipped": 3,
    "failed": 0,
    "seconds": 200.81,
    "measured_at_commit": "144baf4",
}

#: How many persisted games are rebuilt from their identifier alone and compared
#: byte for byte with what is stored.
ISOLATED_REBUILD_GAMES = 200


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


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def verify_prerequisites() -> dict:
    """Agent 1 must be PASS and every frozen identity must still be live."""
    contract_path = DATA_DIRECTORY / "agent_01_warmstart_contract.json"
    agent_one = json.loads(contract_path.read_text())
    gates = agent_one.get("completion_gates", {})
    upstream_problems = wc.verify_frozen_upstream(include_library_digest=True)
    roster_problems = wc.verify_teacher_roster()
    population_problems = rp.verify_live_population()
    live_digest = wc.contract_digest()
    return {
        "agent_01_status": agent_one.get("status"),
        "agent_01_gates_true": bool(gates) and all(gates.values()),
        "agent_01_contract_digest": agent_one.get("contract_digest"),
        "live_contract_digest": live_digest,
        "contract_digest_matches": live_digest == agent_one.get("contract_digest"),
        "upstream_problems": upstream_problems,
        "roster_problems": roster_problems,
        "population_problems": population_problems,
        "policy_roster_digest": rp.roster_digest(),
        "prerequisites_met": (
            agent_one.get("status") == "PASS"
            and bool(gates)
            and all(gates.values())
            and live_digest == agent_one.get("contract_digest")
            and not upstream_problems
            and not roster_problems
            and not population_problems
        ),
    }


# ---------------------------------------------------------------------------
# Independent determinism evidence at production scale
# ---------------------------------------------------------------------------


def isolated_rebuild_audit(root: Path, games: int = ISOLATED_REBUILD_GAMES) -> dict:
    """Rebuild persisted games from their identifier alone and compare bytes.

    The strongest determinism evidence available: a game is replayed from
    scratch through the same frozen path the generator used, re-encoded, and its
    payload digest compared with the digest the commit journal recorded when the
    corpus was written.
    """
    reader = CorpusReader(root, CORPUS_SPLITS)
    game_ids = reader.game_ids()
    if not game_ids:
        return {"rebuilt": 0, "problems": ["the corpus is empty"]}
    step = max(1, len(game_ids) // max(1, games))
    sample = game_ids[::step][:games]
    problems: list[str] = []
    started = time.perf_counter()
    for game_id in sample:
        commit = reader.commits[game_id]
        stored_metadata = reader.metadata(game_id)
        rebuilt = rp.play_corpus_game(game_id)
        payload = compress(encode_game_record(rebuilt.record), DEFAULT_COMPRESSION_LEVEL)
        if payload_digest(payload) != commit.trajectory_sha256:
            problems.append(f"{game_id}: rebuilt trajectory bytes differ from the stored ones")
        if rebuilt.metadata != stored_metadata:
            problems.append(f"{game_id}: rebuilt metadata differs from the stored record")
    return {
        "rebuilt": len(sample),
        "problems": problems,
        "seconds": time.perf_counter() - started,
    }


def crash_resume_evidence(games: int = 6) -> dict:
    """A clean and a crash-interrupted mini-corpus, finalized side by side.

    Run here as well as in the regression suite so the accepted artifact records
    the property on this machine, at this revision, with these frozen contracts.
    """
    from stratego.training.warmstart_seed import synthetic_game_id

    game_ids = tuple(
        synthetic_game_id("train", "tactical_rule_based@1.0.0", "basic_heuristic@1.0.0", index)
        for index in range(games)
    )
    workspace = Path(tempfile.mkdtemp(prefix="phase8_agent02_crash_"))
    try:
        clean = workspace / "clean"
        crashed = workspace / "crashed"
        sc.generate_corpus(clean, worker_count=1, chunks_per_worker=1, game_ids=game_ids)

        class _Interrupt(RuntimeError):
            pass

        seen = {"commits": 0}

        def hook(stage: str, _writer) -> None:
            if stage == "after_commit":
                seen["commits"] += 1
            elif stage == "after_metadata" and seen["commits"] == 2:
                raise _Interrupt("interrupted between metadata and commit")

        interrupted = False
        try:
            sc.generate_games(crashed, game_ids, segment=0, worker_id=0, crash_hook=hook)
        except _Interrupt:
            interrupted = True
        sc.generate_corpus(crashed, worker_count=2, chunks_per_worker=1, game_ids=game_ids)

        clean_digest = corpus_content_digest(clean, CORPUS_SPLITS)
        crashed_digest = corpus_content_digest(crashed, CORPUS_SPLITS)
        clean_audit = sc.audit_corpus(clean, worker_count=1, observation_plies=2)
        crashed_audit = sc.audit_corpus(crashed, worker_count=1, observation_plies=2)
        return {
            "games": games,
            "interrupted": interrupted,
            "clean_content_digest": clean_digest,
            "crashed_content_digest": crashed_digest,
            "digests_agree": clean_digest == crashed_digest,
            "clean_problems": clean_audit["problems"],
            "crashed_problems": crashed_audit["problems"],
            "crashed_integrity": {
                key: value
                for key, value in crashed_audit["integrity"].items()
                if key
                in (
                    "committed_count",
                    "duplicate_committed_ids",
                    "orphan_trajectory_records",
                    "orphan_metadata_records",
                    "missing_trajectory_payloads",
                    "missing_metadata_records",
                )
            },
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def regeneration_evidence(
    reference_digest: str, *, workers: int, chunks_per_worker: int, scratch: "str | None"
) -> dict:
    """Rebuild the whole corpus from scratch elsewhere and compare digests.

    The regeneration instructions in the manifest claim a fresh run reproduces
    this corpus. This runs that claim: the entire 28,000-game schedule is
    generated into a throwaway root and its content digest is compared with the
    production one. It also measures generation throughput and peak RSS, which
    is why the numbers reported for generation come from here after a
    finalize-only pass — an identical run, proven identical by the digest.

    The production corpus is never touched; the scratch root is removed at the
    end whether or not the digests agree.
    """
    workspace = Path(tempfile.mkdtemp(prefix="phase8_agent02_regen_", dir=scratch))
    try:
        generated = sc.generate_corpus(
            workspace / "corpus",
            worker_count=workers,
            chunks_per_worker=chunks_per_worker,
        )
        digest = corpus_content_digest(workspace / "corpus", CORPUS_SPLITS)
        return {
            "reference_content_digest": reference_digest,
            "regenerated_content_digest": digest,
            "digests_agree": digest == reference_digest,
            "games_generated": generated["games_generated"],
            "decisions_generated": generated["decisions_generated"],
            "wall_clock_seconds": generated["wall_clock_seconds"],
            "games_per_second": generated["games_per_second"],
            "decisions_per_second": generated["decisions_per_second"],
            "worker_count": generated["worker_count"],
            "chunks": len(generated["chunks"]),
            "memory": generated["memory"],
            "bytes": generated["bytes"],
            "seconds_by_phase": generated["seconds_by_phase"],
            "scratch_root": str(workspace),
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def worker_independence_evidence(games: int = 8) -> dict:
    """The same games generated serially and in parallel, digests compared."""
    from stratego.training.warmstart_seed import synthetic_game_id

    game_ids = tuple(
        synthetic_game_id("validation", "stress_chaos@1.0.0", "strategic_rule_based@1.1.0", index)
        for index in range(games)
    )
    workspace = Path(tempfile.mkdtemp(prefix="phase8_agent02_workers_"))
    try:
        serial = workspace / "serial"
        parallel = workspace / "parallel"
        sc.generate_corpus(serial, worker_count=1, chunks_per_worker=1, game_ids=game_ids)
        sc.generate_corpus(
            parallel, worker_count=4, chunks_per_worker=2, game_ids=tuple(reversed(game_ids))
        )
        serial_digest = corpus_content_digest(serial, CORPUS_SPLITS)
        parallel_digest = corpus_content_digest(parallel, CORPUS_SPLITS)
        return {
            "games": games,
            "serial_digest": serial_digest,
            "parallel_reversed_digest": parallel_digest,
            "digests_agree": serial_digest == parallel_digest,
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def decision_totals(audit: dict) -> dict:
    """Decision and selected-decision totals per split, from the cell table."""
    per_split: dict = {}
    for key, cell in audit["cells"].items():
        split = key.split("|")[0]
        entry = per_split.setdefault(
            split,
            {"games": 0, "plies": 0, "decisions": 0, "selected_decisions": 0},
        )
        entry["games"] += cell["games"]
        entry["plies"] += cell["plies"]
        entry["decisions"] += cell["decisions"]
        entry["selected_decisions"] += cell["selected_decisions"]
    for entry in per_split.values():
        entry["mean_plies"] = round(entry["plies"] / entry["games"], 3) if entry["games"] else 0.0
        entry["mean_selected_decisions"] = (
            round(entry["selected_decisions"] / entry["games"], 3) if entry["games"] else 0.0
        )
    totals = {
        "games": sum(entry["games"] for entry in per_split.values()),
        "plies": sum(entry["plies"] for entry in per_split.values()),
        "decisions": sum(entry["decisions"] for entry in per_split.values()),
        "selected_decisions": sum(entry["selected_decisions"] for entry in per_split.values()),
    }
    return {"per_split": per_split, "totals": totals}


def color_balance(audit: dict) -> dict:
    """Aggregate red/blue/draw counts per split — a diagnostic, not a gate."""
    per_split: dict = {}
    for key, cell in audit["cells"].items():
        split = key.split("|")[0]
        entry = per_split.setdefault(split, {"red_wins": 0, "blue_wins": 0, "draws": 0})
        entry["red_wins"] += cell["red_wins"]
        entry["blue_wins"] += cell["blue_wins"]
        entry["draws"] += cell["draws"]
    for entry in per_split.values():
        total = entry["red_wins"] + entry["blue_wins"] + entry["draws"]
        entry["games"] = total
        entry["red_win_rate"] = round(entry["red_wins"] / total, 4) if total else 0.0
        entry["blue_win_rate"] = round(entry["blue_wins"] / total, 4) if total else 0.0
        entry["draw_rate"] = round(entry["draws"] / total, 4) if total else 0.0
    return per_split


def write_matchup_csv(rows: list) -> None:
    MATCHUP_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with MATCHUP_ARTIFACT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sc.MATCHUP_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [".venv/bin/python", "-m", "pytest", "tests", "-q"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    numbers = {
        name: int(value)
        for value, name in re.findall(r"(\d+) (passed|failed|skipped|error[s]?)", tail)
    }
    return {
        "command": ".venv/bin/python -m pytest tests -q",
        "summary": tail,
        "passed": numbers.get("passed", 0),
        "skipped": numbers.get("skipped", 0),
        "failed": numbers.get("failed", 0) + numbers.get("errors", 0) + numbers.get("error", 0),
        "seconds": round(elapsed, 2),
        "returncode": completed.returncode,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="corpus root; defaults to synthetic_corpus.default_corpus_root()",
    )
    parser.add_argument("--generate", action="store_true", help="generate/resume the corpus")
    parser.add_argument("--finalize", action="store_true", help="audit and write artifacts")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--audit-workers", type=int, default=10)
    parser.add_argument("--chunks-per-worker", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="stop after N new games")
    parser.add_argument("--observation-plies", type=int, default=4)
    parser.add_argument(
        "--measure-regeneration",
        action="store_true",
        help="regenerate the whole corpus into a scratch root and compare digests",
    )
    parser.add_argument("--scratch", default=None, help="parent directory for scratch roots")
    parser.add_argument("--run-pytest", action="store_true")
    arguments = parser.parse_args()

    if not (arguments.generate or arguments.finalize):
        parser.error("choose --generate, --finalize, or both")

    root = Path(arguments.root) if arguments.root else sc.default_corpus_root()
    durations: dict = {}
    started_all = time.perf_counter()
    print(f"corpus root: {root}  ({sc.describe_corpus_root()['source']})")

    started = time.perf_counter()
    prerequisites = verify_prerequisites()
    durations["prerequisites"] = time.perf_counter() - started
    print(f"prerequisites met: {prerequisites['prerequisites_met']}")
    if not prerequisites["prerequisites_met"]:
        print(json.dumps(prerequisites, indent=2))
        return 2

    commands = []
    generation = None
    if arguments.generate:
        started = time.perf_counter()
        generation = sc.generate_corpus(
            root,
            worker_count=arguments.workers,
            chunks_per_worker=arguments.chunks_per_worker,
            limit=arguments.limit,
        )
        durations["generation"] = time.perf_counter() - started
        commands.append(
            f"python scripts/run_phase8_agent02.py --generate --workers {arguments.workers}"
        )
        print(
            f"generated {generation['games_generated']} games "
            f"({generation.get('games_per_second', 0):.1f} games/s, "
            f"{generation.get('decisions_per_second', 0):.0f} decisions/s)"
        )

    if not arguments.finalize:
        print(json.dumps({"generation": generation}, indent=2, default=str)[:2000])
        return 0

    commands.append("python scripts/run_phase8_agent02.py --finalize")

    started = time.perf_counter()
    final = sc.finalize_corpus(
        root,
        worker_count=arguments.audit_workers,
        observation_plies=arguments.observation_plies,
        full_provenance_games=None,
        generation_commands=commands,
    )
    durations["finalize"] = time.perf_counter() - started
    audit = final["audit"]
    manifest = final["manifest"]
    print(
        f"audited {audit['audited_games']} games, "
        f"{audit['replayed_decisions']} decisions, "
        f"{len(audit['problems'])} problems"
    )

    started = time.perf_counter()
    rebuild = isolated_rebuild_audit(root)
    durations["isolated_rebuild"] = time.perf_counter() - started

    started = time.perf_counter()
    crash = crash_resume_evidence()
    workers = worker_independence_evidence()
    durations["determinism_evidence"] = time.perf_counter() - started

    regeneration = None
    if arguments.measure_regeneration:
        started = time.perf_counter()
        regeneration = regeneration_evidence(
            manifest["content_digest"],
            workers=arguments.workers,
            chunks_per_worker=arguments.chunks_per_worker,
            scratch=arguments.scratch,
        )
        durations["regeneration"] = time.perf_counter() - started
        commands.append(
            "python scripts/run_phase8_agent02.py --finalize --measure-regeneration"
        )
        print(
            f"regeneration digests agree: {regeneration['digests_agree']} "
            f"({regeneration['games_per_second']:.1f} games/s)"
        )

    rows = sc.matchup_rows(audit)
    write_matchup_csv(rows)

    gates = sc.completion_gates(audit)
    gates["isolated_rebuild_exact"] = not rebuild["problems"]
    gates["crash_resume_converges"] = bool(
        crash["digests_agree"]
        and crash["interrupted"]
        and not crash["crashed_problems"]
        and not crash["crashed_integrity"]["orphan_trajectory_records"]
        and not crash["crashed_integrity"]["orphan_metadata_records"]
    )
    gates["worker_and_order_independent"] = bool(workers["digests_agree"])
    gates["manifest_and_digests_written"] = bool(manifest["content_digest"])
    if regeneration is not None:
        gates["from_scratch_regeneration_identical"] = bool(regeneration["digests_agree"])

    tests_after = run_pytest() if arguments.run_pytest else None
    if tests_after is not None:
        gates["full_suite_green"] = tests_after["failed"] == 0 and tests_after["returncode"] == 0
        durations["pytest"] = tests_after["seconds"]

    status = "PASS" if all(gates.values()) else "FAIL"
    totals = decision_totals(audit)

    corpus_manifest_path = root / "manifest.json"
    corpus_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    common = {
        "phase": 8,
        "agent": 2,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_revision": _git("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        **_environment(),
        "prerequisite_versions": {
            "warmstart_training_contract": wc.WARMSTART_TRAINING_CONTRACT_VERSION,
            "synthetic_corpus": manifest["corpus_version"],
            "commit_protocol": CORPUS_COMMIT_VERSION,
            "rule_population": rp.RULE_POPULATION_VERSION,
            "trajectory": manifest["trajectory_schema"],
            "setup_library": manifest["setup_library_version"],
            "setup_sampler": manifest["sampler_version"],
            "setup_source": manifest["setup_source_version"],
        },
        "prerequisite_digests": {
            "agent_01_contract": prerequisites["live_contract_digest"],
            "setup_library": manifest["setup_library_digest"],
            "policy_roster": manifest["policy_roster_digest"],
        },
        "tests_before": TESTS_BEFORE,
        "tests_after": tests_after,
        "commands": commands,
        "durations": {key: round(value, 3) for key, value in durations.items()},
        "total_seconds": round(time.perf_counter() - started_all, 3),
        "seeds": {"corpus_master_seed": manifest["corpus_master_seed"]},
    }

    manifest_artifact = {
        **common,
        "artifact": "agent_02_corpus_manifest",
        "files_created": [
            sc.repository_relative(corpus_manifest_path),
            "reports/phase_8_data/agent_02_corpus_manifest.json",
            "reports/phase_8_data/agent_02_corpus_audit.json",
            "reports/phase_8_data/agent_02_matchup_counts.csv",
            "stratego/training/rule_population.py",
            "stratego/training/corpus_commit.py",
            "stratego/training/synthetic_corpus.py",
            "tests/training/test_synthetic_corpus.py",
            "tests/training/test_corpus_resume.py",
            "scripts/run_phase8_agent02.py",
        ],
        "files_modified": [".gitignore"],
        "corpus_manifest": manifest,
        "generation": generation,
        "regeneration": regeneration,
        "storage": audit["storage"],
        "decision_totals": totals,
        "completion_gates": gates,
        "problems": audit["problems"][:50],
        "deviations": [],
    }
    MANIFEST_ARTIFACT.write_text(json.dumps(manifest_artifact, indent=2, sort_keys=True) + "\n")

    audit_artifact = {
        **common,
        "artifact": "agent_02_corpus_audit",
        "prerequisites": prerequisites,
        "audit": audit,
        "decision_totals": totals,
        "color_balance": color_balance(audit),
        "isolated_rebuild": rebuild,
        "crash_resume_evidence": crash,
        "worker_independence_evidence": workers,
        "regeneration_evidence": regeneration,
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "handoff_to_agent_3": {
            "corpus_root": str(root.resolve()),
            "manifest": str(corpus_manifest_path.resolve()),
            "content_digest": manifest["content_digest"],
            "metadata_digest": manifest["metadata_digest"],
            "commit_index_digest": manifest["commit_index_digest"],
            "reader": "stratego.training.corpus_commit.CorpusReader(root, CORPUS_SPLITS)",
            "game_index": "CorpusReader.game_ids(split) — ascending, journal-backed",
            "trajectory_reader": "CorpusReader.record(game_id) -> trajectory_v1 GameRecord",
            "metadata_reader": "CorpusReader.metadata(game_id) -> synthetic sidecar",
            "commit_index_reader": "CorpusReader.commits[game_id] -> CommitRecord",
            "rebuild_game_api": "stratego.training.rule_population.play_corpus_game(game_id)",
            "split_access": "stratego.training.warmstart_contract.corpus_setup_source(split)",
            "policy_weights": dict(wc.POLICY_SUPERVISION_WEIGHTS),
            "decision_sampler": (
                "stratego.training.warmstart_seed.selected_decision_indices(game_id, "
                "total_decisions) — warmstart_decision_sampler_v1"
            ),
            "selected_examples": totals["totals"]["selected_decisions"],
            "audit_apis": [
                "stratego.training.synthetic_corpus.replay_game",
                "stratego.training.synthetic_corpus.audit_provenance",
                "stratego.training.corpus_commit.audit_commit_integrity",
            ],
        },
        "problems": audit["problems"][:200],
        "deviations": [],
    }
    AUDIT_ARTIFACT.write_text(json.dumps(audit_artifact, indent=2, sort_keys=True) + "\n")

    print(f"status: {status}  gates {sum(1 for v in gates.values() if v)}/{len(gates)}")
    for name, value in sorted(gates.items()):
        if not value:
            print(f"  FAILED GATE: {name}")
    print(f"content digest: {manifest['content_digest']}")
    print(f"selected training decisions: {totals['totals']['selected_decisions']}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
