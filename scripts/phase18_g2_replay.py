#!/usr/bin/env python3
"""Phase 18 Gate G2: deterministic replay of the frozen landscape and the assay
endpoints, after the three seeds have run.

This is a separate verification tool so the sources the launch manifest
digested stay untouched. It re-derives, from the frozen documents and seeds
alone:

* the landscape (table digest, exact optimum with its duality certificate,
  the uniform baseline moments) from `phase18_g2_synthetic_landscape_v1.json`;
* the frozen design digest from the code;
* per seed: the initial raw model from its model seed (digest), the first
  period's pool and every one of its outcomes from the seeds (against the
  outcome receipts and the recorded period digest), the initial EMA
  evaluation (against `utilities_initial.npy`), the final EMA evaluation
  from the three-object checkpoint (against `utilities_final.npy`), and
  every period's outcome digest from the receipts.

A mismatch is reported, never repaired.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from stratego.training.phase18.setup_contract import file_sha256, json_document_digest  # noqa: E402
from stratego.training.phase18.setup_learning import SetupTrainer  # noqa: E402
from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest  # noqa: E402
from stratego.training.phase18.setup_sampling import generate_pool  # noqa: E402
from stratego.training.phase18.synthetic_assay import AssayDesign, evaluate_policy  # noqa: E402
from stratego.training.phase18.synthetic_landscape import exact_optimum, landscape_from_document, uniform_moments  # noqa: E402

CANONICAL_ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
REPORTS = CANONICAL_ROOT / "reports" / "phase18"
ARTIFACTS = CANONICAL_ROOT / "artifacts" / "phase18" / "g2_setup_parity_v1"
SYMBOL = {-1: "-", 0: "0", 1: "+"}


def log(message: str) -> None:
    print(f"[g2-replay {time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    import torch

    started = time.perf_counter()
    contract = json.loads((REPORTS / "phase18_g2_contract_v1.json").read_text())
    landscape_document = json.loads((REPORTS / "phase18_g2_synthetic_landscape_v1.json").read_text())
    launch = json.loads((REPORTS / "phase18_g2_launch_manifest_v1.json").read_text())
    design = AssayDesign(namespace=contract["design"]["namespace"], run_id=contract["run_id"])
    torch.set_num_threads(design.threads)

    landscape = landscape_from_document(landscape_document)
    rebuilt = landscape.document()
    moments = uniform_moments(landscape.table)
    optimum = exact_optimum(landscape.table)
    landscape_check = {
        "table_digest_matches": rebuilt["table_digest"] == landscape_document["table_digest"],
        "document_digest_matches": json_document_digest(rebuilt) == json_document_digest(landscape_document),
        "optimum_matches": abs(optimum["optimum"] - landscape_document["exact_optimum"]["optimum"]) < 1e-9,
        "optimum_certified": optimum["certificate"]["certified"],
        "uniform_mean_matches": abs(moments["mean"] - landscape_document["uniform_baseline"]["mean"]) < 1e-12,
        "uniform_sd_matches": abs(moments["sd"] - landscape_document["uniform_baseline"]["sd"]) < 1e-12,
        "reflection_invariant": bool(rebuilt["reflection_invariant"]),
    }
    design_check = {"design_digest_matches_launch": json_document_digest(design.document()) == launch["design_digest"]}
    log(f"landscape replay: {landscape_check}")

    seeds: dict = {}
    for k in design.seed_indices:
        directory = ARTIFACTS / f"seed_{k}"
        record = json.loads((directory / "seed_result.json").read_text())
        check: dict = {}

        model = build_setup_model(device=design.device, seed=design.model_seed(k))
        check["initial_raw_digest_matches"] = state_dict_digest(model) == record["initial_raw_digest"]

        receipts = [json.loads(line) for line in (directory / "outcome_receipts.jsonl").read_text().splitlines()]
        check["receipts_sha256_matches"] = file_sha256(directory / "outcome_receipts.jsonl") == record["outcome_receipts"]["sha256"]
        check["telemetry_sha256_matches"] = file_sha256(directory / "telemetry.jsonl") == record["telemetry"]["sha256"]
        by_period: dict = {}
        for row in receipts:
            by_period.setdefault(row["period"], []).append(row)
        period_digests = [json_document_digest([[r["fingerprint"], r["outcomes"]] for r in by_period[p]]) for p in sorted(by_period)]
        check["all_period_digests_match"] = period_digests == record["period_outcome_digests"]
        check["periods"] = len(by_period)

        first = generate_pool(model, namespace=design.namespace, seed_index=k, snapshot_iteration=0, snapshot_digest=state_dict_digest(model), count=design.pool_size, device=design.device)
        recorded = {row["index"]: row for row in by_period[1]}
        replayed_rows = 0
        fingerprint_mismatches = 0
        outcome_mismatches = 0
        for sample in first.samples:
            if not sample.opening_move:
                continue
            row = recorded.get(sample.index)
            if row is None or row["fingerprint"] != sample.content_fingerprint:
                fingerprint_mismatches += 1
                continue
            outcomes = landscape.outcomes_for(sample.played_canonical, seed_index=k, period=1, fingerprint=sample.content_fingerprint, replicates=design.outcomes_per_setup)
            if "".join(SYMBOL[z] for z in outcomes) != row["outcomes"]:
                outcome_mismatches += 1
            replayed_rows += 1
        check["period_1_rows_replayed"] = replayed_rows
        check["period_1_fingerprint_mismatches"] = fingerprint_mismatches
        check["period_1_outcome_mismatches"] = outcome_mismatches
        check["period_1_replays_exactly"] = replayed_rows == len(by_period[1]) and fingerprint_mismatches == 0 and outcome_mismatches == 0

        initial = evaluate_policy(model.eval(), landscape, design, k, "replay_initial", ema_updates=0)
        stored_initial = np.load(directory / "utilities_initial.npy")
        check["initial_evaluation_max_abs_diff"] = float(np.abs(initial["utilities"] - stored_initial).max())
        check["initial_evaluation_replays"] = bool(np.allclose(initial["utilities"], stored_initial, atol=1e-9))
        check["initial_ema_digest_matches"] = initial["ema_digest"] == record["initial"]["ema_digest"]

        trainer, manifest = SetupTrainer.load_checkpoint(directory / "checkpoint_final", design.training_config(), namespace=design.namespace, seed_index=k, device=design.device)
        final = evaluate_policy(trainer.evaluation_model(device=design.device), landscape, design, k, "replay_final", ema_updates=trainer.ema.updates)
        stored_final = np.load(directory / "utilities_final.npy")
        check["final_evaluation_max_abs_diff"] = float(np.abs(final["utilities"] - stored_final).max())
        check["final_evaluation_replays"] = bool(np.allclose(final["utilities"], stored_final, atol=1e-9))
        check["final_ema_digest_matches"] = final["ema_digest"] == record["final"]["ema_digest"] == manifest["ema"]["state_digest"]
        check["checkpoint_ema_updates"] = trainer.ema.updates
        check["checkpoint_optimizer_steps"] = trainer.optimizer_step_count
        boolean_checks = (
            "initial_raw_digest_matches", "receipts_sha256_matches", "telemetry_sha256_matches",
            "all_period_digests_match", "period_1_replays_exactly", "initial_evaluation_replays",
            "initial_ema_digest_matches", "final_evaluation_replays", "final_ema_digest_matches",
        )
        check["boolean_checks"] = list(boolean_checks)
        check["all"] = all(bool(check[name]) for name in boolean_checks)
        seeds[str(k)] = check
        log(f"seed {k}: replay {'OK' if check['all'] else 'MISMATCH'} - period 1 rows {replayed_rows}, initial diff {check['initial_evaluation_max_abs_diff']:.2e}, final diff {check['final_evaluation_max_abs_diff']:.2e}")

    record = {
        "artifact": "phase18_g2_replay_v1",
        "work_package": contract["work_package"],
        "agent": "phase_18_agent_4",
        "gate": "G2",
        "run_id": contract["run_id"],
        "g2_source_commit": launch["source"]["g2_source_commit"],
        "contract_sha256": file_sha256(REPORTS / "phase18_g2_contract_v1.json"),
        "landscape_sha256": file_sha256(REPORTS / "phase18_g2_synthetic_landscape_v1.json"),
        "landscape": landscape_check,
        "design": design_check,
        "seeds": seeds,
        "all_replays_exact": all(landscape_check.values()) and design_check["design_digest_matches_launch"] and all(s["all"] for s in seeds.values()),
        "note": "a new verification script; it re-derives the frozen landscape, the initial models, the first period's pool and outcomes, and both evaluation endpoints from the frozen seeds and the three-object checkpoints. It is not part of the launch-manifest digests because it was written after the launch and changes nothing the assay executed through.",
        "seconds": round(time.perf_counter() - started, 3),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = REPORTS / "g2" / "phase18_g2_replay_v1.json"
    out.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    log(f"replay {'EXACT' if record['all_replays_exact'] else 'MISMATCH'}; written to {out}")
    return 0 if record["all_replays_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
