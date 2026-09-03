#!/usr/bin/env python3
"""Phase 18 Agent 6, Stage 6A replacement design: the supporting tables for the matched
two-lineage joint-bundle pilot (candidate with setup learning against a control whose setup
model stays frozen at its initial version), reproducibly. Analytic only; no Monte Carlo (the resampling unit was
validated in scripts/phase18_g3_stage6a_tables.py, T5, and that validation does not depend on
the number of cases per base).

Reads tracked evidence only, through the Stage 6A constants loader plus:

* reports/phase18/phase18_g1_control_run_v1.json      C1 supervised update cost (25,000 updates) and batch size
* reports/phase_8_data/agent_02_corpus_manifest.json   selected decisions per teacher game, trajectory bytes per game

No game is played, no pool sampled, no model built, no held-out material read.

Usage
-----
    python scripts/phase18_g3_joint_design_tables.py --write
    python scripts/phase18_g3_joint_design_tables.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/phase18/g3_design/phase18_g3_joint_design_tables_v2.json"

_spec = importlib.util.spec_from_file_location("stage6a_tables", ROOT / "scripts/phase18_g3_stage6a_tables.py")
S6A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S6A)

C = S6A.C
V = S6A.V
SD_PAIRED = S6A.SD_PAIRED
MARGIN = S6A.MARGIN
Z975, Z80 = S6A.Z975, S6A.Z80
SEC_PER_GAME = S6A.SEC_PER_GAME
MEAN_PLIES = S6A.MEAN_PLIES
POOL = S6A.POOL
SLOTS = S6A.SLOTS_PROVISIONAL
T_PLIES = S6A.PLIES_PER_PERIOD
RETENTION = S6A.RETENTION_PERIODS
G1_GPS = C["G1_HARNESS_GAMES_PER_SECOND"][0]

# Frozen structure of the matched-bundle evaluation (operator decisions 2026-09-02, correction round)
BASES = 160                      # 10 per family, the screening slice base_index 400..409
FAMILIES = 16
HANDCRAFTED_OPPONENTS = ("basic_heuristic", "strategic_rule_based", "tactical_rule_based", "stress_scout_rush",
                         "stress_miner_rush", "stress_berserker", "stress_information_miser", "stress_chaos")
LINEAGES = 2                     # candidate (setup updates enabled) and control (setup model frozen at its initial version)
COLOURS = 2
CASES_PER_BASE = len(HANDCRAFTED_OPPONENTS) * COLOURS
CASES_PER_ARM = BASES * CASES_PER_BASE

# Declared estimates with no tracked source (each is cited in the design as an estimate)
OUTCOME_RECEIPT_BYTES = S6A.OUTCOME_RECEIPT_BYTES
SETUP_CHECKPOINT_MB = S6A.CHECKPOINT_MB
C1_WEIGHTS_MB = C["C0_CHECKPOINT_BYTES"][0] / 1e6
C1_PARAMS = 863_959
C1_OPTIMIZER_MB = 2 * C1_PARAMS * 4 / 1e6            # AdamW first and second moments, float32
SLOT_STATE_KB = 2.0                                  # engine state per active slot, estimate
BUFFER_ROW_KB = 0.5                                  # buffer row (tokens, masks, log-probs), estimate


def _find_key(obj, key: str, source: str, path: str = ""):
    """First occurrence of `key` in a nested JSON document, with the path it was found at."""
    if isinstance(obj, dict):
        if key in obj and not isinstance(obj[key], (dict, list)):
            return obj[key], f"{source}:{path}/{key}"
        for k, v in obj.items():
            found = _find_key(v, key, source, f"{path}/{k}")
            if found is not None:
                return found
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found = _find_key(v, key, source, f"{path}[{i}]")
            if found is not None:
                return found
    return None


def load_extra() -> dict:
    run = json.load(open(ROOT / "reports/phase18/phase18_g1_control_run_v1.json"))["run"]
    updates = run["updates_completed"]
    wall = run["wall_seconds"]
    validation = run["validation_seconds"]
    manifest = json.load(open(ROOT / "reports/phase_8_data/agent_02_corpus_manifest.json"))
    dt = manifest["decision_totals"]["per_split"]["train"]
    st = manifest["storage"]
    return {
        "C1_SECONDS_PER_UPDATE": ((wall - validation) / updates, "reports/phase18/phase18_g1_control_run_v1.json: (run.wall_seconds - run.validation_seconds) / run.updates_completed"),
        "C1_UPDATES_MEASURED": (updates, "same file, run.updates_completed"),
        "C1_BATCH_SIZE": _find_key(json.load(open(ROOT / "reports/phase18/phase18_g1_control_run_v1.json")), "batch_size",
                                   "reports/phase18/phase18_g1_control_run_v1.json"),
        "SELECTED_DECISIONS_PER_GAME": (dt["mean_selected_decisions"], "reports/phase_8_data/agent_02_corpus_manifest.json: decision_totals.per_split.train.mean_selected_decisions"),
        "TRAJECTORY_BYTES_PER_GAME": (st["total_bytes"] / manifest["decision_totals"]["totals"]["games"], "reports/phase_8_data/agent_02_corpus_manifest.json: storage.total_bytes / decision_totals.totals.games"),
    }


X = load_extra()
C1_SEC = X["C1_SECONDS_PER_UPDATE"][0]
SEL_PER_GAME = X["SELECTED_DECISIONS_PER_GAME"][0]
TRAJ_BYTES = X["TRAJECTORY_BYTES_PER_GAME"][0]


def ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def table_period() -> dict:
    completions = SLOTS * T_PLIES / MEAN_PLIES
    rows = []
    for K in (32, 64, 128, 256):
        rows.append({"c1_updates_per_period": K, "c1_seconds": K * C1_SEC,
                     "collector_cpu_seconds": completions * SEC_PER_GAME,
                     **{f"wall_s_at_{r}gps": completions / r + K * C1_SEC for r in (30, 60, 100)}})
    return {"completions_per_period": completions, "outcomes_per_period": 2 * completions,
            "outcomes_per_setup": 2 * completions / POOL, "live_examples_per_period": completions * SEL_PER_GAME,
            "trajectory_MB_per_period": completions * TRAJ_BYTES / 1e6, "rows": rows}


def table_budget() -> dict:
    completions = SLOTS * T_PLIES / MEAN_PLIES
    rows = []
    for periods in (64, 128, 256):
        for K in (64, 128):
            games = completions * periods
            cpu_h = (games * SEC_PER_GAME + periods * K * C1_SEC) / 3600.0
            rows.append({"periods": periods, "c1_updates_per_period": K, "games_per_lineage": games,
                         "c1_updates_per_lineage": periods * K, "live_examples_per_lineage": games * SEL_PER_GAME,
                         "cpu_hours_per_lineage": cpu_h, "cpu_hours_per_pilot": LINEAGES * cpu_h,
                         **{f"wall_h_per_pilot_at_{r}gps": LINEAGES * (games / r + periods * K * C1_SEC) / 3600.0 for r in (30, 60, 100)},
                         "trajectory_GB_per_pilot": LINEAGES * games * TRAJ_BYTES / 1e9,
                         "outcome_receipts_MB_per_pilot": LINEAGES * games * 2 * OUTCOME_RECEIPT_BYTES / 1e6})
    return {"lineages": LINEAGES, "seeds": 1, "rows": rows}


def table_bundle() -> dict:
    collector_mb = (SLOTS * SLOT_STATE_KB + RETENTION * POOL * BUFFER_ROW_KB) / 1e3
    total = C1_WEIGHTS_MB + C1_OPTIMIZER_MB + SETUP_CHECKPOINT_MB + collector_mb
    return {"c1_weights_MB": C1_WEIGHTS_MB, "c1_optimizer_MB_estimate": C1_OPTIMIZER_MB, "setup_raw_opt_ema_MB": SETUP_CHECKPOINT_MB,
            "collector_state_MB_estimate": collector_mb, "bundle_MB_estimate": total,
            "bundles_per_256_periods_every_32": 256 // 32, "bundle_storage_MB_per_256_periods": total * (256 // 32)}


ARM_SETS = (
    ("primary: candidate_final vs control_final", 2),
    ("primary + candidate diagnostics (candidate_128, candidate_0)", 4),
    ("primary + candidate diagnostics + control_128 (optional)", 5),
)


def table_evaluation() -> dict:
    rows = []
    for seeds in (1, 2):                 # one seed; a second seed only as conditional follow-up
        for label, arms in ARM_SETS:
            games = arms * CASES_PER_ARM * seeds
            rows.append({"seeds": seeds, "arm_set": label, "arms": arms, "cases_per_arm": CASES_PER_ARM,
                         "games": games, "hours_g1_harness": games / G1_GPS / 3600.0,
                         "paired_games_per_opponent": COLOURS * BASES * seeds,
                         "paired_games_per_opponent_colour": BASES * seeds,
                         "paired_games_per_family": CASES_PER_BASE * BASES * seeds // FAMILIES})
    return {"bases": BASES, "opponents": list(HANDCRAFTED_OPPONENTS), "colours": COLOURS, "cases_per_base": CASES_PER_BASE,
            "cases_per_arm": CASES_PER_ARM, "g1_harness_games_per_second": G1_GPS, "rows": rows}


def var_factor(rho_b, rho_w, seeds):
    rest = 1.0 - rho_b - rho_w
    return rho_b + rho_w / CASES_PER_BASE + rest / (CASES_PER_BASE * seeds)


def pass_prob(d: float, se: float) -> float:
    return 1.0 - ncdf(max(Z975 - d / se, (MARGIN - d) / se))


def table_resolution() -> dict:
    rows = []
    for seeds in (1, 2):
        for rho_b in (0.0, 0.05, 0.10, 0.20):
            f = var_factor(rho_b, 0.10, seeds)
            se = math.sqrt(V * f / BASES)
            rows.append({"seeds": seeds, "rho_b": rho_b, "rho_w": 0.10, "per_base_sd": math.sqrt(V * f), "n_eff": BASES / f,
                         "se": se, "half_width": Z975 * se, "mde80_lower_gt_0": (Z975 + Z80) * se,
                         "d80_combined_rule": max(MARGIN + Z80 * se, (Z975 + Z80) * se),
                         "pass_prob_at_0.05": pass_prob(0.05, se), "pass_prob_at_0.08": pass_prob(0.08, se),
                         "pass_prob_at_0.10": pass_prob(0.10, se)})
    return {"model": "Var(pooled candidate-minus-control contrast) = (V / B) * [rho_b + rho_w / C + (1 - rho_b - rho_w) / (C * seeds)], "
                     f"C = {CASES_PER_BASE} cases per base, one independent lineage pair per seed, type effects shared across seeds (worst case); "
                     "V is the frozen paired per-game variance, which may understate a contrast whose two arms also differ in mover",
            "stratified_bootstrap": "bases resampled with replacement within each family, carrying every opponent, colour and "
                                    "arm of the base; rescaling sqrt(n_f / (n_f - 1)) with n_f = 10; validated in Stage 6A T5",
            "near_boundary_rule": "the 95% interval of the candidate-minus-control contrast contains 0.05; the second seed is then a conditional follow-up",
            "rows": rows}


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render(t: dict) -> str:
    L = []
    L.append("## J1 Constants used by the joint design (in addition to Stage 6A T3)\n")
    L.append("| constant | value | source |")
    L.append("|---|---|---|")
    for k, (v, s) in t["extra_constants"].items():
        L.append(f"| {k} | {fmt(v)} | {s} |")
    for k in ("SECONDS_PER_TEACHER_GAME", "MEAN_PLIES_TRAINING_RULES", "G1_HARNESS_GAMES_PER_SECOND", "C0_CHECKPOINT_BYTES"):
        L.append(f"| {k} | {fmt(C[k][0])} | {C[k][1]} |")
    p = t["period"]
    L.append(f"\n## J2 Joint period cost: S = {SLOTS} slots (provisional) x T = {T_PLIES} plies; {p['completions_per_period']:.0f} completions, {p['outcomes_per_period']:.0f} outcomes ({p['outcomes_per_setup']:.2f} per setup), {p['live_examples_per_period']:.0f} live examples, {p['trajectory_MB_per_period']:.1f} MB of trajectories per period\n")
    L.append("| C1 updates per period | C1 seconds | collector CPU s | wall s @30 games/s | @60 | @100 |")
    L.append("|---|---|---|---|---|---|")
    for r in p["rows"]:
        L.append(f"| {r['c1_updates_per_period']} | {r['c1_seconds']:.1f} | {r['collector_cpu_seconds']:.0f} | {r['wall_s_at_30gps']:.0f} | {r['wall_s_at_60gps']:.0f} | {r['wall_s_at_100gps']:.0f} |")
    L.append(f"\n## J3 Bounded pilot budget: two lineages (candidate and control), one seed (cost only; example building and memory are measured in the pilot)\n")
    L.append("| periods | C1 updates/period | games per lineage | C1 updates per lineage | live examples per lineage | CPU h per lineage | CPU h per pilot | wall h per pilot @30 | @60 | @100 | trajectories GB per pilot | receipts MB per pilot |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in t["budget"]["rows"]:
        L.append(f"| {r['periods']} | {r['c1_updates_per_period']} | {r['games_per_lineage']/1e6:.2f} M | {r['c1_updates_per_lineage']} | {r['live_examples_per_lineage']/1e6:.1f} M | {r['cpu_hours_per_lineage']:.1f} | {r['cpu_hours_per_pilot']:.1f} | {r['wall_h_per_pilot_at_30gps']:.1f} | {r['wall_h_per_pilot_at_60gps']:.1f} | {r['wall_h_per_pilot_at_100gps']:.1f} | {r['trajectory_GB_per_pilot']:.1f} | {r['outcome_receipts_MB_per_pilot']:.0f} |")
    b = t["bundle"]
    L.append(f"\n## J4 Joint bundle size (estimate): C1 weights {b['c1_weights_MB']:.1f} MB + C1 optimizer {b['c1_optimizer_MB_estimate']:.1f} MB + setup raw/optimizer/EMA {b['setup_raw_opt_ema_MB']:.1f} MB + collector state {b['collector_state_MB_estimate']:.1f} MB = {b['bundle_MB_estimate']:.0f} MB; {b['bundles_per_256_periods_every_32']} bundles per 256 periods at every 32 = {b['bundle_storage_MB_per_256_periods']:.0f} MB\n")
    e = t["evaluation"]
    L.append(f"## J5 Candidate-versus-control evaluation on the G1 harness: {e['bases']} bases x {len(e['opponents'])} handcrafted opponents x {e['colours']} colours = {e['cases_per_arm']} cases per arm; {e['g1_harness_games_per_second']:.2f} games/s\n")
    L.append("| seeds | arm set | arms | games | hours (G1 harness) | paired games per opponent | per opponent x colour | per family |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in e["rows"]:
        L.append(f"| {r['seeds']} | {r['arm_set']} | {r['arms']} | {r['games']} | {r['hours_g1_harness']:.2f} | {r['paired_games_per_opponent']} | {r['paired_games_per_opponent_colour']} | {r['paired_games_per_family']} |")
    rs = t["resolution"]
    L.append(f"\n## J6 Resolution of the paired candidate-minus-control contrast at B = {BASES} bases, {CASES_PER_BASE} cases per base (rho_w = 0.10); the near-boundary zone is the half-width around 0.05\n")
    L.append("| seeds | rho_b | per-base SD | n_eff | SE | 95% half-width | MDE (lower>0, 80%) | d80 combined rule | P(pass at 0.05) | P(pass at 0.08) | P(pass at 0.10) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs["rows"]:
        L.append(f"| {r['seeds']} | {r['rho_b']} | {fmt(r['per_base_sd'])} | {r['n_eff']:.0f} | {fmt(r['se'])} | {fmt(r['half_width'])} | {fmt(r['mde80_lower_gt_0'])} | {fmt(r['d80_combined_rule'])} | {fmt(r['pass_prob_at_0.05'],3)} | {fmt(r['pass_prob_at_0.08'],3)} | {fmt(r['pass_prob_at_0.10'],3)} |")
    return "\n".join(L) + "\n"


def build() -> dict:
    return {"extra_constants": {k: list(v) for k, v in X.items()}, "period": table_period(), "budget": table_budget(),
            "bundle": table_bundle(), "evaluation": table_evaluation(), "resolution": table_resolution(),
            "structure": {"bases": BASES, "opponents": list(HANDCRAFTED_OPPONENTS), "colours": COLOURS, "cases_per_arm": CASES_PER_ARM,
                          "lineages": LINEAGES, "seeds": 1, "second_seed": "conditional follow-up only",
                          "slots_provisional": SLOTS, "plies_per_period": T_PLIES, "retention_periods": RETENTION}}


def canonical(t: dict) -> str:
    return json.dumps(t, indent=1, sort_keys=True, allow_nan=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    t = build()
    text = canonical(t)
    if not a.quiet:
        print(render(t), end="")
    digest = hashlib.sha256(text.encode()).hexdigest()
    if a.write:
        OUT.write_text(text + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}  sha256 {digest}")
    if a.check:
        on_disk = OUT.read_text().rstrip("\n")
        ok = on_disk == text
        print(f"check {'OK' if ok else 'MISMATCH'}: recomputed sha256 {digest}; on disk {hashlib.sha256(on_disk.encode()).hexdigest()}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
