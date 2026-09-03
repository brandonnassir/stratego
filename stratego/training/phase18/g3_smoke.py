"""Phase 18 Stage 6B: the tiny smoke configuration and the restart check
(design section 6, the one new end-to-end test).

```text
control    a fresh lineage runs periods 1..n+1 uninterrupted in this process
restart    a fresh lineage runs periods 1..n, writes bundle_n with at least one
           game unfinished; a NEW PROCESS resumes bundle_n and runs period n+1
assert     period n+1 is identical: completed game ids, outcome records, the
           attribution map, the accounting, the live commit digest, the C1
           batch keys, and the digests of C1, setup raw, setup EMA, the buffer
           and the bundle written at n+1
```

Equivalence is proved on CPU with one thread (P18-D002: MPS is not bitwise
reproducible). The smoke configuration is small enough to run inside the
suite and is never a performance claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from ..warmstart_checkpoint import verify_corpus_identity
from ..warmstart_trainer import unit_test_config
from .g3_bundle import read_manifest
from .g3_contract import LINEAGE_CANDIDATE, Phase18G3Error, PilotConfig
from .g3_pilot import BUNDLES_DIRECTORY, LineageRunner, bundle_name, read_period_records

UNIFORM_PRIOR = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

#: Compared field by field between the uninterrupted and the restarted period.
RESTART_COMPARED_FIELDS = (
    ("collection", "started"),
    ("collection", "completed"),
    ("collection", "in_flight_at_end"),
    ("collection", "plies_advanced"),
    ("collection", "outcomes_attributed"),
    ("collection", "cross_period_attributions"),
    ("collection", "completed_game_ids_digest"),
    ("collection", "outcome_records_digest"),
    ("collection", "live", "commit_digest"),
    ("collection", "live", "games"),
    ("c1", "keys_digests"),
    ("c1", "live_universe_digest"),
    ("c1", "cursor_after_planned"),
    ("c1_state_digest",),
    ("c1_global_step",),
    ("setup_raw_digest",),
    ("setup_ema_digest",),
    ("setup_updates",),
    ("setup_optimizer_steps",),
    ("setup_ema_updates",),
    ("buffer", "state_digest"),
    ("buffer", "rows"),
    ("buffer", "outcomes_added"),
    ("integrity",),
)

RESTART_COMPARED_COMPONENTS = ("c1", "setup_raw", "setup_ema")


def smoke_pilot_config(*, lineage: str = LINEAGE_CANDIDATE, namespace: str, run_id: str, overrides: "dict | None" = None) -> PilotConfig:
    """A CPU, one-thread, C0-model configuration that leaves games unfinished."""
    fields = dict(
        run_id=run_id,
        namespace=namespace,
        seed_index=1,
        lineage=lineage,
        c1_train_config=unit_test_config(batch_size=8, model_candidate="C0"),
        canonical_per_batch=4,
        live_per_batch=4,
        periods=3,
        c1_updates_per_period=2,
        slots=3,
        pool_size=8,
        plies_per_period=64,
        # basic/tactical cells: games of roughly 50-150 plies, so some finish
        # inside a period and some span the save point.
        schedule_cells=4,
        cell_indices=(11, 12, 21, 22),
        buffer_storage_periods=101,
        live_retention_periods=2,
        bundle_cadence_periods=1,
        threads=1,
        loader_workers=1,
        loader_prefetch=1,
        record_cache_size=16,
    )
    fields.update(overrides or {})
    return PilotConfig(**fields)


def _runner_keywords(run_root, corpus_root):
    identity = verify_corpus_identity(corpus_root, None, check_payload_bytes=False)
    return dict(run_root=run_root, corpus_root=corpus_root, corpus_identity=identity, value_prior=UNIFORM_PRIOR, require_complete_split=False, log=lambda message: None)


def _lookup(record, path):
    value = record
    for key in path:
        value = value[key]
    return value


def resume_and_run_one_period(*, config: PilotConfig, run_root, corpus_root, bundle_directory) -> dict:
    """The restarted half: resume `bundle_directory` and play exactly one period."""
    torch.set_num_threads(int(config.threads))
    runner = LineageRunner.resume(config, bundle_directory=bundle_directory, **_runner_keywords(run_root, corpus_root))
    try:
        record = runner.run_period()
    finally:
        runner.close()
    return record


def restart_check(*, root, corpus_root, namespace: str, run_id: str = "G3-RESTART-CHECK", restart_after: int = 1, overrides: "dict | None" = None, lineage: str = LINEAGE_CANDIDATE) -> dict:
    """Run the design-section-6 restart test and return its evidence."""
    root = Path(root)
    corpus_root = Path(corpus_root)
    config = smoke_pilot_config(lineage=lineage, namespace=namespace, run_id=run_id, overrides=overrides)
    torch.set_num_threads(int(config.threads))
    n = int(restart_after)
    if not 1 <= n < config.periods:
        raise Phase18G3Error("restart_after must leave at least one period to replay")
    started = time.perf_counter()

    control_root = root / "uninterrupted"
    control = LineageRunner.fresh(config, **_runner_keywords(control_root, corpus_root))
    try:
        control_records = control.run(periods=n + 1)
    finally:
        control.close()

    restart_root = root / "restarted"
    first = LineageRunner.fresh(config, **_runner_keywords(restart_root, corpus_root))
    try:
        first_records = first.run(periods=n)
    finally:
        first.close()
    bundle_n = restart_root / config.lineage / BUNDLES_DIRECTORY / bundle_name(n)
    manifest_n = read_manifest(bundle_n)
    unfinished = int(manifest_n["components"]["collector"]["active_games"])
    if unfinished < 1:
        raise Phase18G3Error("the smoke configuration left no game unfinished at the save point; the test is void")

    command = [
        sys.executable,
        "-m",
        "stratego.training.phase18.g3_smoke",
        "--resume-one-period",
        "--run-root",
        str(restart_root),
        "--corpus-root",
        str(corpus_root),
        "--bundle",
        str(bundle_n),
        "--namespace",
        namespace,
        "--run-id",
        run_id,
        "--lineage",
        lineage,
        "--overrides",
        json.dumps(overrides or {}),
    ]
    completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[3]), capture_output=True, text=True)
    if completed.returncode != 0:
        raise Phase18G3Error(f"the restarted process failed:\n{completed.stdout}\n{completed.stderr}")

    control_next = control_records[n]
    restarted_records = read_period_records(restart_root / config.lineage)
    if len(restarted_records) != n + 1:
        raise Phase18G3Error(f"the restarted lineage recorded {len(restarted_records)} periods, expected {n + 1}")
    restarted_next = restarted_records[n]
    comparisons = {}
    problems = []
    for path in RESTART_COMPARED_FIELDS:
        try:
            same = _lookup(control_next, path) == _lookup(restarted_next, path)
        except KeyError:
            same = False
        comparisons["/".join(path)] = same
        if not same:
            problems.append("/".join(path))
    control_bundle = read_manifest(control_root / config.lineage / BUNDLES_DIRECTORY / bundle_name(n + 1))
    restarted_bundle = read_manifest(restart_root / config.lineage / BUNDLES_DIRECTORY / bundle_name(n + 1))
    bundle_comparison = {}
    for name in RESTART_COMPARED_COMPONENTS:
        same = control_bundle["components"][name]["state_digest"] == restarted_bundle["components"][name]["state_digest"]
        bundle_comparison[name] = same
        if not same:
            problems.append(f"bundle_{n + 1} component {name}")
    same_buffer = control_bundle["components"]["collector"]["buffer_state_digest"] == restarted_bundle["components"]["collector"]["buffer_state_digest"]
    bundle_comparison["buffer_state"] = same_buffer
    if not same_buffer:
        problems.append(f"bundle_{n + 1} buffer state")
    same_counters = control_bundle["counters"] == restarted_bundle["counters"]
    bundle_comparison["counters"] = same_counters
    if not same_counters:
        problems.append(f"bundle_{n + 1} counters")
    # The periods before the restart point were played by the same code path.
    prefix_same = all(
        control_records[i]["collection"]["completed_game_ids_digest"] == first_records[i]["collection"]["completed_game_ids_digest"]
        for i in range(n)
    )
    return {
        "check": "phase18_g3_restart_check_v1",
        "lineage": lineage,
        "restart_after_period": n,
        "unfinished_games_at_save": unfinished,
        "control_next_period_completed": int(control_next["collection"]["completed"]),
        "prefix_identical": prefix_same,
        "comparisons": comparisons,
        "bundle_comparison": bundle_comparison,
        "problems": problems,
        "passed": not problems and prefix_same,
        "restarted_process_returncode": completed.returncode,
        "device": config.c1_train_config.device,
        "threads": int(config.threads),
        "seconds": round(time.perf_counter() - started, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-one-period", action="store_true")
    parser.add_argument("--run-root")
    parser.add_argument("--corpus-root")
    parser.add_argument("--bundle")
    parser.add_argument("--namespace")
    parser.add_argument("--run-id")
    parser.add_argument("--lineage", default=LINEAGE_CANDIDATE)
    parser.add_argument("--overrides", default="{}")
    arguments = parser.parse_args(argv)
    if not arguments.resume_one_period:
        parser.error("only --resume-one-period is a command")
    config = smoke_pilot_config(lineage=arguments.lineage, namespace=arguments.namespace, run_id=arguments.run_id, overrides=json.loads(arguments.overrides))
    record = resume_and_run_one_period(config=config, run_root=Path(arguments.run_root), corpus_root=Path(arguments.corpus_root), bundle_directory=Path(arguments.bundle))
    print(json.dumps({"period": record["period"], "c1_state_digest": record["c1_state_digest"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
