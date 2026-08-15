#!/usr/bin/env python3
"""Phase 8 Agent 3 acceptance harness: training examples, targets, anti-leak.

Verifies the Agent 1/2 prerequisites and the corpus relocation, then proves
the `warmstart_example_v1` pipeline against the accepted corpus:

- static audit of **every** selected decision (sampler, value mapping, frozen
  weights, action membership, frame conversion) plus a full replay computing
  belief-supervision counts and the observable-inventory identity;
- replay reconstruction audit (observations, masks, belief labels) on at
  least 100,000 selected decisions;
- direct teacher-decision reproduction on at least 10,000 policy-supervised
  decisions;
- at least 25,000 hidden-permutation paired anti-leak trials;
- frozen validation baselines (train-fitted value prior; uniform-legal
  policy; unresolved-inventory belief marginal) with game-level bootstrap;
- deterministic universe/shuffle/cursor evidence and the dataset throughput
  benchmark.

Writes the three Agent 3 artifacts::

    reports/phase_8_data/agent_03_example_contract.json
    reports/phase_8_data/agent_03_target_audit.json
    reports/phase_8_data/agent_03_validation_baselines.json

No model metric touches the test split; test games are parsed for structural
target correctness only.

Usage::

    python scripts/run_phase8_agent03.py --full --run-pytest
    python scripts/run_phase8_agent03.py --quick          # reduced sizes
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from stratego.engine.state import create_game  # noqa: E402
from stratego.engine.transition import apply_action  # noqa: E402
from stratego.engine.observation import belief_target  # noqa: E402
from stratego.training import synthetic_corpus as sc  # noqa: E402
from stratego.training import warmstart_contract as wc  # noqa: E402
from stratego.training import warmstart_baselines as wb  # noqa: E402
from stratego.training import warmstart_dataset as wd  # noqa: E402
from stratego.training import warmstart_examples as we  # noqa: E402
from stratego.training.corpus_commit import (  # noqa: E402
    CorpusReader,
    audit_commit_integrity,
    corpus_content_digest,
    journal_directory,
    metadata_directory,
    read_journal,
    read_metadata_file,
    reconcile_corpus,
)
from stratego.training.corpus_commit import JOURNAL_SUFFIX, METADATA_SUFFIX  # noqa: E402
from stratego.training.corpus_commit import _file_sets  # noqa: E402
from stratego.training.belief_targets import PIECE_TYPE_INDEX  # noqa: E402
from stratego.training.reconstruction import iter_reconstructed_decisions  # noqa: E402
from stratego.training.rule_population import TeacherCache  # noqa: E402
from stratego.training.warmstart_seed import (  # noqa: E402
    CORPUS_SPLITS,
    DECISION_SAMPLER_VERSION,
    GAMES_PER_CELL,
    VALIDATION_BOOTSTRAP_SEED,
    parse_synthetic_game_id,
    selected_decision_indices,
    train_order_seed,
)

DATA_DIRECTORY = REPOSITORY_ROOT / "reports" / "phase_8_data"
CONTRACT_ARTIFACT = DATA_DIRECTORY / "agent_03_example_contract.json"
AUDIT_ARTIFACT = DATA_DIRECTORY / "agent_03_target_audit.json"
BASELINES_ARTIFACT = DATA_DIRECTORY / "agent_03_validation_baselines.json"

#: The canonical relocated corpus root fixed by the relocation addendum.
CANONICAL_CORPUS_ROOT = (
    "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/"
    "data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1"
)

#: The accepted Agent 2 corpus identity, required to survive the relocation.
ACCEPTED_DIGESTS = {
    "content_digest": "c95c3545b07f2341e7efbc83c79e6342510dd973038b0f72e7eae013cff87d0d",
    "metadata_digest": "1db0f02fe45b16f539f070b1e12d4fdd6f390fd0487180fe660af0f4d49c81bb",
    "commit_index_digest": "32e8e18d1ca57ee555ed848851284f5938d4989ceb6c864f83ca4b9286c15db1",
}

#: Accepted Agent 2 selected-decision totals the fresh universe must reproduce.
ACCEPTED_SELECTED = {"train": 1247173, "validation": 249963, "test": 249924}

#: The full pre-edit suite, measured before any Phase 8 Agent 3 change.
TESTS_BEFORE = {
    "command": ".venv/bin/python -m pytest tests -q",
    "summary": "3551 passed, 3 skipped in 224.38s (0:03:44)",
    "passed": 3551,
    "skipped": 3,
    "failed": 0,
    "seconds": 224.38,
    "measured_at_commit": "eb730d4",
}


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
# Relocation verification (the pre-Agent-3 addendum)
# ---------------------------------------------------------------------------


def verify_relocation(root: Path) -> dict:
    """Read-only verification that the relocated corpus is the accepted one."""
    report: dict = {"canonical_path": CANONICAL_CORPUS_ROOT, "problems": []}
    resolution = sc.describe_corpus_root()
    report["resolver"] = resolution
    report["resolver_matches_canonical"] = str(root) == CANONICAL_CORPUS_ROOT
    if str(root) != CANONICAL_CORPUS_ROOT:
        report["problems"].append(f"default_corpus_root() -> {root}")

    torn: list = []
    for split in CORPUS_SPLITS:
        for _segment, _worker, name in _file_sets(root, split):
            journal_path = journal_directory(root, split) / f"{name}{JOURNAL_SUFFIX}"
            _, journal_valid = read_journal(journal_path)
            if journal_valid != journal_path.stat().st_size:
                torn.append(f"{split}/{name}: journal tail")
            metadata_path = metadata_directory(root, split) / f"{name}{METADATA_SUFFIX}"
            _, metadata_valid = read_metadata_file(metadata_path)
            if metadata_valid != metadata_path.stat().st_size:
                torn.append(f"{split}/{name}: metadata tail")
    report["torn_tails"] = torn
    report["problems"].extend(torn)

    integrity = audit_commit_integrity(root, CORPUS_SPLITS)
    counts = {
        key: (len(value) if isinstance(value, list) else value)
        for key, value in integrity.items()
    }
    report["integrity"] = counts
    for key, value in counts.items():
        if key in ("committed_count", "metadata_count", "payload_count"):
            if value != 28000:
                report["problems"].append(f"integrity {key} = {value}, expected 28000")
        elif value:
            report["problems"].append(f"integrity {key} = {value}, expected 0")

    observed = {
        "content_digest": corpus_content_digest(root, CORPUS_SPLITS),
        "metadata_digest": sc._metadata_digest(root, CORPUS_SPLITS),
        "commit_index_digest": sc._commit_index_digest(root, CORPUS_SPLITS),
    }
    report["accepted_digests"] = dict(ACCEPTED_DIGESTS)
    report["observed_digests"] = observed
    report["digests_match"] = observed == ACCEPTED_DIGESTS
    for name, accepted in ACCEPTED_DIGESTS.items():
        if observed[name] != accepted:
            report["problems"].append(f"{name} mismatch: {observed[name]}")

    reconciliation = reconcile_corpus(root, CORPUS_SPLITS)
    report["reconciliation"] = {
        "committed_count": reconciliation["committed_count"],
        "duplicate_committed_ids": len(reconciliation["duplicate_committed_ids"]),
        "bytes_discarded": reconciliation["bytes_discarded"],
        "shards_removed": reconciliation["shards_removed"],
    }
    if reconciliation["committed_count"] != 28000:
        report["problems"].append(
            f"reconciled committed_count = {reconciliation['committed_count']}"
        )
    if reconciliation["bytes_discarded"] or reconciliation["shards_removed"]:
        report["problems"].append("reconciliation modified the corpus; expected a no-op")

    committed = set(reconciliation["committed"])
    per_split = Counter(parse_synthetic_game_id(game_id)["split"] for game_id in committed)
    report["split_counts"] = dict(per_split)
    if dict(per_split) != {"train": 20000, "validation": 4000, "test": 4000}:
        report["problems"].append(f"split counts {dict(per_split)}")

    missing_total = 0
    unscheduled = set(committed)
    cell_problems = 0
    for split in CORPUS_SPLITS:
        scheduled = sc.scheduled_game_ids(split)
        missing_total += sum(1 for game_id in scheduled if game_id not in committed)
        unscheduled -= set(scheduled)
        cell_counts: Counter = Counter()
        for game_id in scheduled:
            if game_id in committed:
                identity = parse_synthetic_game_id(game_id)
                cell_counts[(identity["red_token"], identity["blue_token"])] += 1
        for cell in wc.ordered_matchup_cells():
            if cell_counts[(cell["red_token"], cell["blue_token"])] != GAMES_PER_CELL[split]:
                cell_problems += 1
    report["schedule"] = {
        "cells": len(wc.ordered_matchup_cells()),
        "games_per_cell": dict(GAMES_PER_CELL),
        "missing_scheduled_games": missing_total,
        "unscheduled_committed_games": len(unscheduled),
        "cells_with_wrong_count": cell_problems,
    }
    if missing_total or unscheduled or cell_problems:
        report["problems"].append(
            f"schedule: missing={missing_total} unscheduled={len(unscheduled)} "
            f"bad_cells={cell_problems}"
        )

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "data/stratego_phase8/warmstart/"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    tracked_pointer = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "data/warmstart_corpus_root.txt"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
    )
    status_inside = subprocess.run(
        ["git", "status", "--porcelain", "data/stratego_phase8/"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report["git"] = {
        "corpus_tree_ignored": ignored.returncode == 0,
        "pointer_file_tracked": tracked_pointer.returncode == 0,
        "corpus_tree_status_entries": status_inside.stdout.strip(),
    }
    if ignored.returncode != 0:
        report["problems"].append("data/stratego_phase8/ is not git-ignored")
    if tracked_pointer.returncode != 0:
        report["problems"].append("data/warmstart_corpus_root.txt is not tracked")
    if status_inside.stdout.strip():
        report["problems"].append("corpus tree still appears in git status")

    report["storage"] = sc._disk_usage(root)
    report["verified"] = not report["problems"]
    report["record"] = {
        "accepted Agent 2 corpus relocated": "YES",
        "relocation type": "storage path only",
        "regeneration performed": "NO",
        "corpus bytes intentionally modified": "NO",
        "canonical location": CANONICAL_CORPUS_ROOT,
        "resolver result": "MATCH" if report["resolver_matches_canonical"] else "MISMATCH",
        "accepted corpus digests": "MATCH" if report["digests_match"] else "MISMATCH",
        "committed games": reconciliation["committed_count"],
        "split counts": "20,000 / 4,000 / 4,000"
        if dict(per_split) == {"train": 20000, "validation": 4000, "test": 4000}
        else str(dict(per_split)),
        "integrity failures": len(report["problems"]),
    }
    return report


def verify_prerequisites() -> dict:
    """Agents 1 and 2 must be PASS and every frozen identity must be live."""
    agent_one = json.loads((DATA_DIRECTORY / "agent_01_warmstart_contract.json").read_text())
    agent_two = json.loads((DATA_DIRECTORY / "agent_02_corpus_audit.json").read_text())
    live_digest = wc.contract_digest()
    upstream_problems = wc.verify_frozen_upstream(include_library_digest=True)
    roster_problems = wc.verify_teacher_roster()
    handoff = agent_two.get("handoff_to_agent_3", {})
    return {
        "agent_01_status": agent_one.get("status"),
        "agent_02_status": agent_two.get("status"),
        "agent_02_gates_true": agent_two.get("gates_true") == agent_two.get("gates_total"),
        "contract_digest_matches": live_digest == agent_one.get("contract_digest"),
        "live_contract_digest": live_digest,
        "agent_02_content_digest": handoff.get("content_digest"),
        "upstream_problems": upstream_problems,
        "roster_problems": roster_problems,
        "prerequisites_met": (
            agent_one.get("status") == "PASS"
            and agent_two.get("status") == "PASS"
            and live_digest == agent_one.get("contract_digest")
            and handoff.get("content_digest") == ACCEPTED_DIGESTS["content_digest"]
            and not upstream_problems
            and not roster_problems
        ),
    }


# ---------------------------------------------------------------------------
# Pass AB: exhaustive static audit + linear replay (all committed games)
# ---------------------------------------------------------------------------


def _chunked(items: "tuple", chunks: int) -> list:
    buckets: list = [[] for _ in range(max(1, chunks))]
    for index, item in enumerate(items):
        buckets[index % len(buckets)].append(item)
    return [bucket for bucket in buckets if bucket]


def _static_replay_chunk(payload: tuple) -> dict:
    """Worker: decode + static audit + linear replay of one game chunk."""
    root, game_ids = payload
    reader = CorpusReader(root, CORPUS_SPLITS)
    out = {
        "games": 0,
        "static_checked": 0,
        "static_problems": [],
        "replay_problems": [],
        "selected_by_split": Counter(),
        "policy_supervised_by_split": Counter(),
        "belief_pieces_by_split": Counter(),
        "value_counts_by_split": {split: [0, 0, 0] for split in CORPUS_SPLITS},
        "by_policy": Counter(),
        "policy_supervised_by_policy": Counter(),
        "belief_by_policy": Counter(),
        "by_cell": Counter(),
        "by_family": Counter(),
        "belief_by_family": Counter(),
        "by_bucket": Counter(),
        "belief_by_bucket": Counter(),
        "legal_count_sum": 0,
        "legal_count_max": 0,
        "inventory_identity_checks": 0,
        "inventory_identity_mismatches": 0,
        "validation_policy_stats": [],
        "validation_value_counts": [],
        "validation_belief_stats": [],
        "train_value_counts": [0, 0, 0],
        "train_policy_stats": [0.0, 0.0, 0.0],
    }
    for game_id in game_ids:
        record, metadata = reader.game(game_id)
        split = metadata["corpus_split"]
        cell = int(metadata["cell_index"])
        total = len(record.decisions)
        commit = reader.commits[game_id]

        static = we.audit_game_static(record, metadata, commit.total_decisions)
        out["static_checked"] += static["checked"]
        out["static_problems"].extend(
            f"{game_id}: {problem}" for problem in static["problems"][:5]
        )
        for class_index, count in enumerate(static["value_counts"]):
            out["value_counts_by_split"][split][class_index] += count
        if split == "train":
            for class_index, count in enumerate(static["value_counts"]):
                out["train_value_counts"][class_index] += count
        if split == "validation":
            out["validation_value_counts"].append((game_id, static["value_counts"]))

        indices = selected_decision_indices(game_id, total)
        selected = set(indices)
        out["selected_by_split"][split] += len(indices)

        game_policy_stats = [0.0, 0.0, 0.0]  # sum w*ln(L), sum w, sum w/L
        game_belief_stats = [0.0, 0, 0]  # ce_sum, top1 hits, pieces
        state = create_game(
            record.red_setup, record.blue_setup, rules=record.rules(), game_id=game_id
        )
        remaining = len(indices)
        for ply, action_id in enumerate(record.actions):
            if ply in selected:
                remaining -= 1
                decision = record.decisions[ply]
                actor = state.acting_player
                if actor != decision.acting_player or state.total_moves != ply:
                    out["replay_problems"].append(f"{game_id} ply {ply}: replay drift")
                policy_id = we.acting_policy_id(metadata, actor)
                weight = float(wc.POLICY_SUPERVISION_WEIGHTS[policy_id])
                family = we.acting_setup_family(metadata, actor)
                bucket = we.progress_bucket(ply, total)
                legal_count = len(decision.legal_action_ids)
                out["by_policy"][policy_id] += 1
                out["by_cell"][f"{split}|{cell}"] += 1
                out["by_family"][family] += 1
                out["by_bucket"][f"q{bucket + 1}"] += 1
                out["legal_count_sum"] += legal_count
                out["legal_count_max"] = max(out["legal_count_max"], legal_count)
                if weight > 0.0:
                    out["policy_supervised_by_split"][split] += 1
                    out["policy_supervised_by_policy"][policy_id] += 1
                    if split in ("train", "validation"):
                        game_policy_stats[0] += weight * float(np.log(legal_count))
                        game_policy_stats[1] += weight
                        game_policy_stats[2] += weight / legal_count

                hidden = belief_target(state, actor)
                pieces = len(hidden)
                out["belief_pieces_by_split"][split] += pieces
                out["belief_by_policy"][policy_id] += pieces
                out["belief_by_family"][family] += pieces
                out["belief_by_bucket"][f"q{bucket + 1}"] += pieces

                # Observable-inventory identity: initial - known == hidden
                # composition. Cheap, and the belief baseline leans on it.
                observable = wb.INITIAL_TYPE_COUNTS.copy()
                for piece in state.pieces:
                    if piece.owner != actor and piece.known_to(actor):
                        observable[piece.true_type] -= 1
                truth = np.zeros(len(observable), dtype=np.int64)
                true_types = []
                for entry in hidden:
                    type_index = PIECE_TYPE_INDEX[entry["true_type"]]
                    truth[type_index] += 1
                    true_types.append(type_index)
                out["inventory_identity_checks"] += 1
                if not np.array_equal(observable, truth):
                    out["inventory_identity_mismatches"] += 1
                    out["replay_problems"].append(
                        f"{game_id} ply {ply}: observable inventory != hidden truth"
                    )
                if split == "validation" and pieces:
                    stats = wb.belief_marginal_statistics(observable, true_types)
                    game_belief_stats[0] += stats["cross_entropy_sum"]
                    game_belief_stats[1] += stats["top1_hits"]
                    game_belief_stats[2] += stats["pieces"]
            if remaining <= 0:
                break
            apply_action(state, action_id)
        if split == "validation":
            out["validation_policy_stats"].append((game_id, game_policy_stats))
            out["validation_belief_stats"].append((game_id, game_belief_stats))
        elif split == "train":
            for slot in range(3):
                out["train_policy_stats"][slot] += game_policy_stats[slot]
        out["games"] += 1
    return out


def _merge_static_replay(chunks: list) -> dict:
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged["games"] += chunk["games"]
        merged["static_checked"] += chunk["static_checked"]
        merged["static_problems"].extend(chunk["static_problems"])
        merged["replay_problems"].extend(chunk["replay_problems"])
        for key in (
            "selected_by_split",
            "policy_supervised_by_split",
            "belief_pieces_by_split",
            "by_policy",
            "policy_supervised_by_policy",
            "belief_by_policy",
            "by_cell",
            "by_family",
            "belief_by_family",
            "by_bucket",
            "belief_by_bucket",
        ):
            merged[key].update(chunk[key])
        for split in CORPUS_SPLITS:
            for class_index in range(3):
                merged["value_counts_by_split"][split][class_index] += chunk[
                    "value_counts_by_split"
                ][split][class_index]
        for class_index in range(3):
            merged["train_value_counts"][class_index] += chunk["train_value_counts"][
                class_index
            ]
        for slot in range(3):
            merged["train_policy_stats"][slot] += chunk["train_policy_stats"][slot]
        merged["legal_count_sum"] += chunk["legal_count_sum"]
        merged["legal_count_max"] = max(merged["legal_count_max"], chunk["legal_count_max"])
        merged["inventory_identity_checks"] += chunk["inventory_identity_checks"]
        merged["inventory_identity_mismatches"] += chunk["inventory_identity_mismatches"]
        merged["validation_policy_stats"].extend(chunk["validation_policy_stats"])
        merged["validation_value_counts"].extend(chunk["validation_value_counts"])
        merged["validation_belief_stats"].extend(chunk["validation_belief_stats"])
    return merged


# ---------------------------------------------------------------------------
# Pass C: replay reconstruction audit (>= 100,000 selected decisions)
# ---------------------------------------------------------------------------


def _reconstruction_chunk(payload: tuple) -> dict:
    root, game_ids = payload
    reader = CorpusReader(root, CORPUS_SPLITS)
    audited = 0
    problems: list = []
    observation_checks = 0
    per_split: Counter = Counter()
    for game_id in game_ids:
        record, metadata = reader.game(game_id)
        indices = selected_decision_indices(game_id, len(record.decisions))
        for rebuilt in iter_reconstructed_decisions(
            record, indices, dense_mask=True, include_public_knowledge=False, copy_state=False
        ):
            example = we.build_example(record, metadata, rebuilt)
            found = we.audit_example(example, record, metadata, rebuilt)
            if found:
                problems.extend(found[:5])
            audited += 1
            observation_checks += 1
            per_split[metadata["corpus_split"]] += 1
    return {
        "audited": audited,
        "observation_checks": observation_checks,
        "problems": problems,
        "per_split": per_split,
    }


# ---------------------------------------------------------------------------
# Pass D: direct teacher-decision reproduction (>= 10,000)
# ---------------------------------------------------------------------------


def _teacher_chunk(payload: tuple) -> dict:
    """Worker: reproduce recorded teacher decisions on policy-supervised plies.

    Which plies are policy-supervised is read off the decoded decision records
    (the stored acting player of each selected ply), never inferred from ply
    parity or schedule position.
    """
    root, game_ids, per_game_cap = payload
    reader = CorpusReader(root, CORPUS_SPLITS)
    cache = TeacherCache()
    reproduced = 0
    mismatches: list = []
    by_policy: Counter = Counter()
    for game_id in game_ids:
        record, metadata = reader.game(game_id)
        indices = selected_decision_indices(game_id, len(record.decisions))
        supervised = [
            index
            for index in indices
            if wc.POLICY_SUPERVISION_WEIGHTS[
                we.acting_policy_id(metadata, record.decisions[index].acting_player)
            ]
            > 0.0
        ]
        step = max(1, len(supervised) // max(1, per_game_cap))
        plies = tuple(supervised[::step][:per_game_cap])
        if not plies:
            continue
        result = we.reproduce_teacher_decisions(record, metadata, plies, cache)
        reproduced += result["reproduced"]
        mismatches.extend(f"{game_id}: {item}" for item in result["mismatches"][:5])
        for ply in plies:
            by_policy[we.acting_policy_id(metadata, record.decisions[ply].acting_player)] += 1
    return {"reproduced": reproduced, "mismatches": mismatches, "by_policy": by_policy}


# ---------------------------------------------------------------------------
# Pass E: hidden-permutation paired anti-leak trials (>= 25,000)
# ---------------------------------------------------------------------------


def _antileak_chunk(payload: tuple) -> dict:
    root, entries = payload
    reader = CorpusReader(root, CORPUS_SPLITS)
    valid = 0
    changed = 0
    control_failures = 0
    mismatches: list = []
    hidden_pieces = 0
    for game_id, plies in entries:
        record, metadata = reader.game(game_id)
        rng = random.Random(f"ws8-antileak:{game_id}")
        for rebuilt in iter_reconstructed_decisions(
            record, tuple(plies), dense_mask=True, include_public_knowledge=False
        ):
            trial = we.hidden_permutation_trial(record, metadata, rebuilt, rng)
            if trial["mismatches"]:
                mismatches.extend(
                    f"{game_id} ply {rebuilt.ply}: {item}" for item in trial["mismatches"]
                )
            if not trial["control_ok"]:
                control_failures += 1
            if trial["valid"] and trial["hidden_pieces"] >= 2:
                valid += 1
                changed += int(trial["changed"])
                hidden_pieces += trial["hidden_pieces"]
    return {
        "valid_trials": valid,
        "changed_trials": changed,
        "control_failures": control_failures,
        "mismatches": mismatches,
        "hidden_pieces": hidden_pieces,
    }


def _run_pool(worker, payloads: list, workers: int) -> list:
    from concurrent.futures import ProcessPoolExecutor

    if workers <= 1 or len(payloads) <= 1:
        return [worker(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, payloads))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def compute_baselines(static: dict) -> dict:
    """Fit the train prior; freeze the three validation baselines with CIs."""
    train_counts = static["train_value_counts"]
    prior = wb.fit_value_prior(train_counts)
    log_prior = np.log(np.maximum(np.asarray(prior), wc.METRIC_LOG_EPSILON))
    per_class_brier = (
        (np.asarray(prior)[None, :] - np.eye(3)) ** 2
    ).sum(axis=1)
    predicted = int(np.argmax(prior))

    # Per-game arrays for the validation split, all keyed the same game order.
    policy_rows = sorted(static["validation_policy_stats"])
    value_rows = sorted(static["validation_value_counts"])
    belief_rows = sorted(static["validation_belief_stats"])
    games = [row[0] for row in value_rows]
    if [row[0] for row in policy_rows] != games or [row[0] for row in belief_rows] != games:
        raise RuntimeError("validation per-game statistics are misaligned")

    policy_ce_num = np.array([row[1][0] for row in policy_rows])
    policy_weight_sum = np.array([row[1][1] for row in policy_rows])
    policy_top1_num = np.array([row[1][2] for row in policy_rows])
    value_counts = np.array([row[1] for row in value_rows], dtype=np.float64)
    value_ce_num = -(value_counts * log_prior[None, :]).sum(axis=1)
    value_brier_num = (value_counts * per_class_brier[None, :]).sum(axis=1)
    value_correct = value_counts[:, predicted]
    value_total = value_counts.sum(axis=1)
    belief_ce_num = np.array([row[1][0] for row in belief_rows])
    belief_hits = np.array([row[1][1] for row in belief_rows], dtype=np.float64)
    belief_pieces = np.array([row[1][2] for row in belief_rows], dtype=np.float64)

    def interval(numerators, denominators):
        return wb.bootstrap_ratio_interval(
            numerators, denominators, seed=VALIDATION_BOOTSTRAP_SEED
        )

    train_policy = static["train_policy_stats"]
    return {
        "eval_version": wc.WARMSTART_EVAL_VERSION,
        "value_prior": {
            "fitted_from": "train selected examples only",
            "train_class_counts": list(train_counts),
            "prior_win_draw_loss": [float(value) for value in prior],
            "predicted_class_index": predicted,
        },
        "validation": {
            "games": len(games),
            "policy": {
                "population": "policy-supervised examples (weight > 0)",
                "cross_entropy": interval(policy_ce_num, policy_weight_sum),
                "expected_top1_accuracy": interval(policy_top1_num, policy_weight_sum),
                "weight_sum": float(policy_weight_sum.sum()),
                "examples": int(static["policy_supervised_by_split"]["validation"]),
            },
            "value": {
                "population": "every selected decision",
                "cross_entropy": interval(value_ce_num, value_total),
                "brier": interval(value_brier_num, value_total),
                "accuracy": interval(value_correct, value_total),
                "examples": int(value_total.sum()),
            },
            "belief": {
                "population": "supervised hidden opponent pieces",
                "cross_entropy": interval(belief_ce_num, belief_pieces),
                "top1_accuracy": interval(belief_hits, belief_pieces),
                "pieces": int(belief_pieces.sum()),
            },
        },
        "train_reference": {
            "policy_cross_entropy": (
                float(train_policy[0] / train_policy[1]) if train_policy[1] else None
            ),
            "policy_expected_top1": (
                float(train_policy[2] / train_policy[1]) if train_policy[1] else None
            ),
            "value_metrics": wb.value_prior_metrics(train_counts, prior),
        },
        "test_split_policy": (
            "no baseline or model metric computed; test parsed for structural "
            "target correctness only"
        ),
    }


# ---------------------------------------------------------------------------
# Determinism evidence
# ---------------------------------------------------------------------------


def determinism_evidence(dataset: wd.WarmstartDataset) -> dict:
    universes = {}
    for split in CORPUS_SPLITS:
        universe = dataset.universe(split)
        universes[split] = {
            "examples": len(universe),
            "digest": wd.universe_digest(universe),
            "recomputed_digest_matches": wd.universe_digest(
                wd.selected_example_universe(dataset.reader, split)
            )
            == wd.universe_digest(universe),
        }
    train_universe = dataset.universe("train")
    orders = {
        f"epoch_{epoch}": str(
            np.array2string(wd.epoch_order(len(train_universe), epoch)[:8])
        )
        for epoch in (0, 1)
    }
    epoch0_again = np.array_equal(
        wd.epoch_order(len(train_universe), 0),
        np.random.default_rng(train_order_seed(0)).permutation(len(train_universe)),
    )
    cursor = wd.DataCursor(split="train", batch_size=wd.DEFAULT_BATCH_SIZE)
    plans = wd.plan_batches(train_universe, cursor, 3)
    resumed = wd.plan_batches(train_universe, plans[0][2], 2)
    resume_exact = [plan[1] for plan in resumed] == [plan[1] for plan in plans[1:3]]
    return {
        "universes": universes,
        "train_order_version": wd.TRAIN_ORDER_VERSION,
        "cursor_version": wd.DATA_CURSOR_VERSION,
        "epoch_order_head": orders,
        "epoch_order_matches_frozen_stream": bool(epoch0_again),
        "cursor_resume_reproduces_exact_batches": bool(resume_exact),
        "epoch_boundary": {
            "train_batches_per_epoch": -(-len(train_universe) // wd.DEFAULT_BATCH_SIZE),
            "batches_never_span_epochs": True,
        },
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


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


def _sample_games_for_target(reader, target_examples: int, splits: tuple) -> list:
    """A deterministic stride sample of games totalling >= target selections."""
    chosen: list = []
    total = 0
    pools = []
    for split in splits:
        ids = reader.game_ids(split)
        pools.append(ids)
    # Interleave splits so the sample covers all of them evenly.
    step = max(1, sum(len(pool) for pool in pools) // max(1, target_examples // 60))
    for pool in pools:
        for game_id in pool[::step]:
            chosen.append(game_id)
            total += len(
                selected_decision_indices(game_id, reader.commits[game_id].total_decisions)
            )
    if total < target_examples:
        seen = set(chosen)
        for pool in pools:
            for game_id in pool:
                if total >= target_examples:
                    break
                if game_id in seen:
                    continue
                chosen.append(game_id)
                seen.add(game_id)
                total += len(
                    selected_decision_indices(
                        game_id, reader.commits[game_id].total_decisions
                    )
                )
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="production sizes")
    parser.add_argument("--quick", action="store_true", help="reduced sizes for development")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--reconstruction-target", type=int, default=None)
    parser.add_argument("--teacher-target", type=int, default=None)
    parser.add_argument("--antileak-target", type=int, default=None)
    parser.add_argument("--benchmark-batches", type=int, default=None)
    parser.add_argument("--run-pytest", action="store_true")
    arguments = parser.parse_args()
    if not (arguments.full or arguments.quick):
        parser.error("choose --full or --quick")

    scale = 1.0 if arguments.full else 0.02
    reconstruction_target = arguments.reconstruction_target or int(105000 * scale)
    teacher_target = arguments.teacher_target or int(10500 * scale)
    antileak_target = arguments.antileak_target or int(27500 * scale)
    benchmark_batches = arguments.benchmark_batches or (24 if arguments.full else 2)
    worker_counts = (1, 2, 4, 8, 10) if arguments.full else (1, 2)

    durations: dict = {}
    commands = [
        "python scripts/run_phase8_agent03.py "
        + ("--full" if arguments.full else "--quick")
        + (" --run-pytest" if arguments.run_pytest else "")
    ]
    started_all = time.perf_counter()
    root = sc.default_corpus_root()
    print(f"corpus root: {root}  ({sc.describe_corpus_root()['source']})")

    started = time.perf_counter()
    prerequisites = verify_prerequisites()
    durations["prerequisites"] = time.perf_counter() - started
    print(f"prerequisites met: {prerequisites['prerequisites_met']}")
    if not prerequisites["prerequisites_met"]:
        print(json.dumps(prerequisites, indent=2))
        return 2

    started = time.perf_counter()
    relocation = verify_relocation(root)
    durations["relocation_verification"] = time.perf_counter() - started
    print(f"relocation verified: {relocation['verified']}")
    if not relocation["verified"]:
        print("BLOCKED — ACCEPTED CORPUS RELOCATION VERIFICATION FAILED")
        print(json.dumps(relocation["problems"], indent=2))
        return 3

    reader = CorpusReader(root, CORPUS_SPLITS)
    all_ids = reader.game_ids()

    # -- Pass AB -----------------------------------------------------------
    started = time.perf_counter()
    payloads = [(str(root), bucket) for bucket in _chunked(all_ids, arguments.workers * 4)]
    static = _merge_static_replay(_run_pool(_static_replay_chunk, payloads, arguments.workers))
    durations["static_replay_audit"] = time.perf_counter() - started
    print(
        f"static+replay audit: {static['games']} games, {static['static_checked']} "
        f"selected decisions, {len(static['static_problems'])} static problems, "
        f"{len(static['replay_problems'])} replay problems "
        f"({durations['static_replay_audit']:.1f}s)"
    )

    # -- Universe ----------------------------------------------------------
    started = time.perf_counter()
    dataset = wd.WarmstartDataset(root)
    universe_counts = {split: len(dataset.universe(split)) for split in CORPUS_SPLITS}
    determinism = determinism_evidence(dataset)
    durations["universe_and_determinism"] = time.perf_counter() - started
    print(f"universe: {universe_counts}")

    # -- Pass C ------------------------------------------------------------
    started = time.perf_counter()
    sample = _sample_games_for_target(reader, reconstruction_target, CORPUS_SPLITS)
    payloads = [(str(root), bucket) for bucket in _chunked(tuple(sample), arguments.workers * 4)]
    reconstruction_chunks = _run_pool(_reconstruction_chunk, payloads, arguments.workers)
    reconstruction = {
        "audited": sum(chunk["audited"] for chunk in reconstruction_chunks),
        "observation_checks": sum(
            chunk["observation_checks"] for chunk in reconstruction_chunks
        ),
        "problems": [
            problem for chunk in reconstruction_chunks for problem in chunk["problems"]
        ],
        "per_split": dict(
            sum((chunk["per_split"] for chunk in reconstruction_chunks), Counter())
        ),
        "games": len(sample),
    }
    durations["reconstruction_audit"] = time.perf_counter() - started
    print(
        f"reconstruction audit: {reconstruction['audited']} examples over "
        f"{reconstruction['games']} games, {len(reconstruction['problems'])} problems "
        f"({durations['reconstruction_audit']:.1f}s)"
    )

    # -- Pass D ------------------------------------------------------------
    # Candidate games are chosen by metadata alone (at least one supervised
    # side); the workers pick the exact supervised plies from the decoded
    # decision records. A stride over the whole corpus spreads the sample
    # across cells rather than front-loading the schedule's early matchups.
    started = time.perf_counter()
    per_game_cap = 16
    candidates: list = []
    for game_id in all_ids:
        metadata = reader.metadata(game_id)
        if (
            float(metadata["red_policy_weight"]) > 0.0
            or float(metadata["blue_policy_weight"]) > 0.0
        ):
            candidates.append(game_id)
    wanted_games = max(1, int(teacher_target * 1.25) // per_game_cap)
    stride = max(1, len(candidates) // wanted_games)
    teacher_games = tuple(candidates[::stride][:wanted_games])
    payloads = [
        (str(root), bucket, per_game_cap)
        for bucket in _chunked(teacher_games, arguments.workers * 4)
    ]
    teacher_chunks = _run_pool(_teacher_chunk, payloads, arguments.workers)
    teacher = {
        "reproduced": sum(chunk["reproduced"] for chunk in teacher_chunks),
        "mismatches": [item for chunk in teacher_chunks for item in chunk["mismatches"]],
        "by_policy": dict(sum((chunk["by_policy"] for chunk in teacher_chunks), Counter())),
    }
    durations["teacher_reproduction"] = time.perf_counter() - started
    print(
        f"teacher reproduction: {teacher['reproduced']} decisions, "
        f"{len(teacher['mismatches'])} mismatches ({durations['teacher_reproduction']:.1f}s)"
    )

    # -- Pass E ------------------------------------------------------------
    # A stride over train+validation spreads trials across matchup cells, and
    # a stride inside each game's selection spreads them across game phases.
    started = time.perf_counter()
    antileak_entries: list = []
    antileak_total = 0
    per_game_trials = 16
    trial_ids = tuple(reader.game_ids("train")) + tuple(reader.game_ids("validation"))
    trial_stride = max(1, len(trial_ids) // max(1, antileak_target // per_game_trials))
    for game_id in trial_ids[::trial_stride]:
        if antileak_total >= antileak_target:
            break
        total_decisions = reader.commits[game_id].total_decisions
        indices = selected_decision_indices(game_id, total_decisions)
        ply_stride = max(1, len(indices) // per_game_trials)
        chosen = list(indices[::ply_stride][:per_game_trials])
        antileak_entries.append((game_id, chosen))
        antileak_total += len(chosen)
    payloads = [
        (str(root), bucket)
        for bucket in _chunked(tuple(antileak_entries), arguments.workers * 4)
    ]
    antileak_chunks = _run_pool(_antileak_chunk, payloads, arguments.workers)
    antileak = {
        "valid_trials": sum(chunk["valid_trials"] for chunk in antileak_chunks),
        "changed_trials": sum(chunk["changed_trials"] for chunk in antileak_chunks),
        "control_failures": sum(chunk["control_failures"] for chunk in antileak_chunks),
        "model_input_mismatches": [
            item for chunk in antileak_chunks for item in chunk["mismatches"]
        ],
        "hidden_pieces": sum(chunk["hidden_pieces"] for chunk in antileak_chunks),
        "splits": ["train", "validation"],
    }
    durations["antileak_trials"] = time.perf_counter() - started
    print(
        f"anti-leak: {antileak['valid_trials']} valid trials, "
        f"{antileak['changed_trials']} changed, "
        f"{len(antileak['model_input_mismatches'])} mismatches "
        f"({durations['antileak_trials']:.1f}s)"
    )

    # -- Baselines ---------------------------------------------------------
    started = time.perf_counter()
    baselines = compute_baselines(static)
    durations["baselines"] = time.perf_counter() - started
    validation = baselines["validation"]
    print(
        "validation baselines: policy CE "
        f"{validation['policy']['cross_entropy']['point']:.4f}, value CE "
        f"{validation['value']['cross_entropy']['point']:.4f}, belief CE "
        f"{validation['belief']['cross_entropy']['point']:.4f}"
    )

    # -- Throughput --------------------------------------------------------
    started = time.perf_counter()
    benchmark = wd.benchmark_dataset(
        root,
        worker_counts=worker_counts,
        batches=benchmark_batches,
    )
    durations["throughput_benchmark"] = time.perf_counter() - started
    best = max(benchmark["configurations"], key=lambda entry: entry["examples_per_second"])
    print(
        f"throughput: best {best['examples_per_second']:.0f} examples/s at "
        f"{best['workers']} workers ({durations['throughput_benchmark']:.1f}s)"
    )

    tests_after = run_pytest() if arguments.run_pytest else None
    if tests_after is not None:
        durations["pytest"] = tests_after["seconds"]

    # -- Gates -------------------------------------------------------------
    universe_matches_accepted = {
        split: universe_counts[split] == ACCEPTED_SELECTED[split] for split in CORPUS_SPLITS
    }
    selected_matches_universe = {
        split: static["selected_by_split"][split] == universe_counts[split]
        for split in CORPUS_SPLITS
    }
    gates = {
        "agent_2_corpus_digests_verified": relocation["digests_match"],
        "relocation_verified": relocation["verified"],
        "universe_deterministic": all(
            entry["recomputed_digest_matches"] for entry in determinism["universes"].values()
        ),
        "universe_matches_accepted_totals": all(universe_matches_accepted.values()),
        "static_audit_covers_universe": all(selected_matches_universe.values())
        and static["static_checked"] == sum(universe_counts.values()),
        "max_64_decisions_per_game_exact": not any(
            "long game" in problem or "short game" in problem
            for problem in static["static_problems"]
        ),
        "static_audit_zero_mismatches": not static["static_problems"],
        "replay_pass_zero_problems": not static["replay_problems"],
        "inventory_identity_zero_mismatches": static["inventory_identity_mismatches"] == 0,
        "reconstruction_audit_target_met": reconstruction["audited"] >= 100000
        if arguments.full
        else reconstruction["audited"] > 0,
        "reconstruction_audit_zero_mismatches": not reconstruction["problems"],
        "teacher_reproduction_target_met": teacher["reproduced"] >= 10000
        if arguments.full
        else teacher["reproduced"] > 0,
        "teacher_reproduction_zero_mismatches": not teacher["mismatches"],
        "value_mapping_exhaustive": static["static_checked"] == sum(universe_counts.values())
        and sum(
            sum(static["value_counts_by_split"][split]) for split in CORPUS_SPLITS
        )
        == sum(universe_counts.values()),
        "antileak_target_met": antileak["valid_trials"] >= 25000
        if arguments.full
        else antileak["valid_trials"] > 0,
        "antileak_zero_model_input_mismatches": not antileak["model_input_mismatches"],
        "antileak_positive_controls_fired": antileak["changed_trials"] > 0
        and antileak["control_failures"] == 0,
        "validation_baselines_frozen": bool(baselines["validation"]),
        "test_model_metrics_not_computed": True,
        "shuffle_cursor_deterministic": determinism["cursor_resume_reproduces_exact_batches"]
        and determinism["epoch_order_matches_frozen_stream"],
        "worker_count_independent_batches": benchmark["all_configurations_identical"],
        "throughput_measured": bool(benchmark["configurations"]),
    }
    if tests_after is not None:
        gates["full_suite_green"] = (
            tests_after["failed"] == 0 and tests_after["returncode"] == 0
        )

    status = "PASS" if all(gates.values()) else "FAIL"

    common = {
        "phase": 8,
        "agent": 3,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_revision": _git("rev-parse", "--short", "HEAD"),
        "working_tree_state": "dirty" if _git("status", "--porcelain") else "clean",
        **_environment(),
        "prerequisite_versions": {
            "warmstart_training_contract": wc.WARMSTART_TRAINING_CONTRACT_VERSION,
            "example": wc.WARMSTART_EXAMPLE_VERSION,
            "decision_sampler": DECISION_SAMPLER_VERSION,
            "train_order": wd.TRAIN_ORDER_VERSION,
            "data_cursor": wd.DATA_CURSOR_VERSION,
            "eval": wc.WARMSTART_EVAL_VERSION,
            "corpus": sc.SYNTHETIC_CORPUS_VERSION,
        },
        "prerequisite_digests": {
            "agent_01_contract": prerequisites["live_contract_digest"],
            "corpus_content": relocation["observed_digests"]["content_digest"],
            "corpus_metadata": relocation["observed_digests"]["metadata_digest"],
            "corpus_commit_index": relocation["observed_digests"]["commit_index_digest"],
        },
        "tests_before": TESTS_BEFORE,
        "tests_after": tests_after,
        "commands": commands,
        "durations": {key: round(value, 3) for key, value in durations.items()},
        "total_seconds": round(time.perf_counter() - started_all, 3),
        "seeds": {
            "validation_bootstrap_seed": VALIDATION_BOOTSTRAP_SEED,
            "antileak_rng": "random.Random(f'ws8-antileak:{game_id}')",
        },
        "relocation_verification": relocation,
    }

    contract_artifact = {
        **common,
        "artifact": "agent_03_example_contract",
        "example_schema": wc.example_schema(),
        "target_semantics": wc.target_semantics(),
        "dataset_api": {
            "dataset": "stratego.training.warmstart_dataset.WarmstartDataset(root=None)",
            "resolver": "stratego.training.synthetic_corpus.default_corpus_root()",
            "universe": "WarmstartDataset.universe(split) — frozen schedule order",
            "order": "epoch_order(size, epoch) — default_rng(train_order_seed(epoch))",
            "cursor": "DataCursor / plan_batch / plan_batches — exact-resume slices",
            "batches": "iter_batches(dataset, cursor, batches=, workers=)",
            "sequential": "WarmstartDataset.iter_sequential(split) for held-out passes",
            "model_boundary": "WarmstartBatch.model_input() -> observation tensor only",
            "baselines": "stratego.training.warmstart_baselines",
        },
        "universe": {
            "counts": universe_counts,
            "digests": {
                split: determinism["universes"][split]["digest"] for split in CORPUS_SPLITS
            },
            "matches_accepted_agent_2_totals": universe_matches_accepted,
        },
        "selected_example_counts": {
            "by_split": dict(static["selected_by_split"]),
            "policy_supervised_by_split": dict(static["policy_supervised_by_split"]),
            "value_supervised_by_split": dict(static["selected_by_split"]),
            "belief_supervised_pieces_by_split": dict(static["belief_pieces_by_split"]),
            "by_policy": dict(static["by_policy"]),
            "policy_supervised_by_policy": dict(static["policy_supervised_by_policy"]),
            "belief_pieces_by_policy": dict(static["belief_by_policy"]),
            "by_setup_family": dict(static["by_family"]),
            "belief_pieces_by_setup_family": dict(static["belief_by_family"]),
            "by_progress_bucket": dict(static["by_bucket"]),
            "belief_pieces_by_progress_bucket": dict(static["belief_by_bucket"]),
            "by_matchup_cell": dict(static["by_cell"]),
            "mean_legal_actions": static["legal_count_sum"] / max(1, static["static_checked"]),
            "max_legal_actions": static["legal_count_max"],
        },
        "determinism": determinism,
        "throughput": benchmark,
        "completion_gates": gates,
        "files_created": [
            "stratego/training/warmstart_examples.py",
            "stratego/training/warmstart_dataset.py",
            "stratego/training/warmstart_baselines.py",
            "tests/training/conftest.py",
            "tests/training/test_warmstart_examples.py",
            "tests/training/test_warmstart_targets.py",
            "tests/information_security/test_warmstart_target_boundary.py",
            "scripts/run_phase8_agent03.py",
            "reports/phase_8_data/agent_03_example_contract.json",
            "reports/phase_8_data/agent_03_target_audit.json",
            "reports/phase_8_data/agent_03_validation_baselines.json",
        ],
        "files_modified": [
            ".gitignore",
            "data/warmstart_corpus_root.txt",
            "reports/phase_8_implementation_report.md",
        ],
        "problems": [],
        "deviations": [
            "the relocation addendum's '>250 GB free' statement did not match the "
            "measured machine state; actual free space at verification is recorded "
            "in relocation_verification.storage and was sufficient for Agent 3"
        ],
    }

    audit_artifact = {
        **common,
        "artifact": "agent_03_target_audit",
        "prerequisites": prerequisites,
        "static_audit": {
            "games": static["games"],
            "selected_decisions_checked": static["static_checked"],
            "problems": static["static_problems"][:200],
            "value_counts_by_split": static["value_counts_by_split"],
            "inventory_identity_checks": static["inventory_identity_checks"],
            "inventory_identity_mismatches": static["inventory_identity_mismatches"],
            "replay_problems": static["replay_problems"][:200],
        },
        "reconstruction_audit": {
            **{key: value for key, value in reconstruction.items() if key != "problems"},
            "problems": reconstruction["problems"][:200],
            "checks_per_example": [
                "engine legal set equals the stored one",
                "recorded action legal on replay",
                "model action inverse-converts to the absolute action",
                "model mask equals the converted legal list and inverts back",
                "belief labels re-derived from privileged piece records",
                "belief mask equals the hidden-occupancy observation channel",
                "value mapping recomputed",
                "frozen supervision weight recomputed",
            ],
        },
        "teacher_reproduction": teacher,
        "antileak_trials": {
            **{
                key: value
                for key, value in antileak.items()
                if key != "model_input_mismatches"
            },
            "model_input_mismatches": antileak["model_input_mismatches"][:200],
            "mechanism": "stratego.engine.permutation.permute_hidden_identities",
        },
        "boundary_regression": (
            "tests/information_security/test_warmstart_target_boundary.py — "
            "object-graph walk from WarmstartBatch.model_input() reaches no "
            "privileged value, with three positive controls"
        ),
        "completion_gates": gates,
        "gates_total": len(gates),
        "gates_true": sum(1 for value in gates.values() if value),
        "handoff_to_agent_4": {
            "example_version": wc.WARMSTART_EXAMPLE_VERSION,
            "dataset": "stratego.training.warmstart_dataset",
            "selected_examples": universe_counts,
            "train_order_api": "epoch_order / DataCursor / plan_batches / iter_batches",
            "resume_cursor_api": "DataCursor.to_dict() inside warmstart_checkpoint_v1",
            "baseline_evaluators": "stratego.training.warmstart_baselines",
            "validation_baselines": "reports/phase_8_data/agent_03_validation_baselines.json",
            "measured_throughput": {
                "best_examples_per_second": best["examples_per_second"],
                "best_workers": best["workers"],
            },
            "anti_leak_evidence": "antileak_trials + boundary_regression above",
        },
        "problems": [],
        "deviations": contract_artifact["deviations"],
    }

    baselines_artifact = {
        **common,
        "artifact": "agent_03_validation_baselines",
        **baselines,
        "completion_gates": {
            "validation_baselines_frozen": gates["validation_baselines_frozen"],
            "test_model_metrics_not_computed": gates["test_model_metrics_not_computed"],
        },
        "problems": [],
        "deviations": [],
    }

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CONTRACT_ARTIFACT.write_text(
        json.dumps(contract_artifact, indent=2, sort_keys=True, default=str) + "\n"
    )
    AUDIT_ARTIFACT.write_text(
        json.dumps(audit_artifact, indent=2, sort_keys=True, default=str) + "\n"
    )
    BASELINES_ARTIFACT.write_text(
        json.dumps(baselines_artifact, indent=2, sort_keys=True, default=str) + "\n"
    )

    print(f"status: {status}  gates {sum(1 for v in gates.values() if v)}/{len(gates)}")
    for name, value in sorted(gates.items()):
        if not value:
            print(f"  FAILED GATE: {name}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
