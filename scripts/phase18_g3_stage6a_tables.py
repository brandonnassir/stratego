#!/usr/bin/env python3
"""Phase 18 Agent 6, Stage 6A: every numerical table in the G3 analysis, reproducibly.

Reads tracked evidence only:

* data/setups/setup_library_v1.jsonl                 (library concentration references)
* reports/phase18/g2/phase18_g2_seed_{1,2,3}_result_v1.json
* reports/phase18/g2_raw_confirmation/phase18_g2_raw_confirmation_seed_{1,2,3}_result_v1.json
                                                      (fresh-model pool baselines, EMA/raw gap)

Planning constants are the frozen ones from reports/phase18/phase18_evaluation_contract_v1.json
(paired per-game difference SD 0.5391, single-lane SD 0.4236, tie fraction 0.594) and the
Phase 4 baseline league (0.087 s per rule-vs-rule game). Nothing here reads a G3 outcome:
no game is played, no pool is sampled, no model is built.

Usage
-----
    python scripts/phase18_g3_stage6a_tables.py --write   # print tables, write the JSON
    python scripts/phase18_g3_stage6a_tables.py --check   # recompute and compare with the JSON

The Monte Carlo sections use numpy's PCG64 with a fixed seed, so the JSON is byte-stable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/phase18/g3_design/phase18_g3_stage6a_tables_v1.json"

# --------------------------------------------------------------------------------------
# Frozen planning inputs (cited in the analysis; never re-estimated here)
# --------------------------------------------------------------------------------------
SD_PAIRED = 0.5391          # own-setup instrument: sd of joint - move_only per board (n = 3,000)
SD_SINGLE_LANE = 0.4236     # single-lane game score sd
TIE_FRACTION = 0.594
V = SD_PAIRED ** 2          # variance of one paired per-game difference
SIGMA2 = SD_SINGLE_LANE ** 2
RHO_CASE = 1.0 - V / (2.0 * SIGMA2)   # implied same-case cross-arm correlation
Z975 = 1.959963984540054
Z80 = 0.8416212335729143
Z90 = 1.2815515655446004
MARGIN = 0.05
EMA_DECAY = 0.999
SECONDS_PER_TEACHER_GAME = 0.087      # Phase 4 baseline league, 44,544 rule-vs-rule games, single process
MEAN_PLIES_TRAINING_RULES = 258.8     # Phase 8 train corpus, battleless 100
OPPONENTS = 10
COLOURS = 2
CASES_PER_BASE = OPPONENTS * COLOURS
SEEDS = 3
FAMILIES = 16
POOL = 1024
RNG_SEED = 20260902


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------------------
# T1  Library concentration references
# --------------------------------------------------------------------------------------
def setup_stats(setups: list[str]) -> dict:
    n = len(setups)
    flag_rank = Counter()
    flag_file = Counter()
    bomb_file = Counter()
    adj = []
    front_bombs = []
    back_bombs = []
    for s in setups:
        i = s.index("F")
        r, f = divmod(i, 10)
        flag_rank[r] += 1
        flag_file[f] += 1
        nb = 0
        for dr, df in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, ff = r + dr, f + df
            if 0 <= rr < 4 and 0 <= ff < 10 and s[rr * 10 + ff] == "B":
                nb += 1
        adj.append(nb)
        front_bombs.append(sum(1 for k in range(30, 40) if s[k] == "B"))
        back_bombs.append(sum(1 for k in range(0, 10) if s[k] == "B"))
        for k, ch in enumerate(s):
            if ch == "B":
                bomb_file[k % 10] += 1
    sym_flag = max((flag_file[f] + flag_file[9 - f]) / (2.0 * n) for f in range(5))
    sym_bomb = max((bomb_file[f] + bomb_file[9 - f]) / (2.0 * 6 * n) for f in range(5))
    return {
        "n": n,
        "front_row_flag_share": flag_rank[3] / n,
        "back_rank_flag_share": flag_rank[0] / n,
        "max_flag_file_share_symmetrized": sym_flag,
        "bombs_adjacent_to_flag_mean": sum(adj) / n,
        "front_row_bomb_share": sum(front_bombs) / (10.0 * n),
        "back_rank_bomb_share": sum(back_bombs) / (10.0 * n),
        "max_bomb_file_share_symmetrized": sym_bomb,
    }


STAT_KEYS = [
    "front_row_flag_share",
    "back_rank_flag_share",
    "max_flag_file_share_symmetrized",
    "bombs_adjacent_to_flag_mean",
    "front_row_bomb_share",
    "back_rank_bomb_share",
    "max_bomb_file_share_symmetrized",
]


def table_library() -> dict:
    rows = [json.loads(line) for line in open(ROOT / "data/setups/setup_library_v1.jsonl")]
    overall = setup_stats([r["canonical_setup"] for r in rows])
    by_family: dict[str, list[str]] = {}
    for r in rows:
        by_family.setdefault(r["family_key"], []).append(r["canonical_setup"])
    fam = {k: setup_stats(v) for k, v in sorted(by_family.items())}
    family_range = {
        key: {
            "min": min(fam[k][key] for k in fam),
            "max": max(fam[k][key] for k in fam),
            "argmin": min(fam, key=lambda k: fam[k][key]),
            "argmax": max(fam, key=lambda k: fam[k][key]),
        }
        for key in STAT_KEYS
    }
    return {"overall": overall, "per_family": fam, "family_range": family_range}


# --------------------------------------------------------------------------------------
# T2  Fresh-model pool baselines from the tracked G2 result files
# --------------------------------------------------------------------------------------
def table_g2_pools() -> dict:
    files = [ROOT / f"reports/phase18/g2/phase18_g2_seed_{s}_result_v1.json" for s in (1, 2, 3)]
    files += [
        ROOT / f"reports/phase18/g2_raw_confirmation/phase18_g2_raw_confirmation_seed_{s}_result_v1.json"
        for s in (1, 2, 3)
    ]
    samples = []
    gaps = []
    for path in files:
        d = json.load(open(path))
        blocks = {
            "ema_initial": d["initial"]["generation_telemetry"],
            "ema_final": d["final"]["generation_telemetry"],
            "raw_initial": d["raw_diagnostic"]["initial"]["generation_telemetry"],
            "raw_final": d["raw_diagnostic"]["final"]["generation_telemetry"],
        }
        for label, t in blocks.items():
            played = t["flag_file_histogram_played"]
            n = t["count"]
            sym = max((played[f] + played[9 - f]) / (2.0 * n) for f in range(5))
            asym = max(
                abs(played[f] - played[9 - f]) / math.sqrt(max(1, played[f] + played[9 - f]))
                for f in range(5)
            )
            samples.append(
                {
                    "file": path.name,
                    "block": label,
                    "count": n,
                    "distinct_class_fraction": t["distinct_class_fingerprints"] / n,
                    "max_flag_file_share_played": max(played) / n,
                    "max_flag_file_share_symmetrized": sym,
                    "max_mirror_asymmetry_z": asym,
                    "mean_sequence_information_nats": t["mean_sequence_information_nats"],
                    "reflected_fraction": t["reflected_fraction"],
                    "immediately_terminal_count": t["immediately_terminal_count"],
                    "legality_failures": t["legality_failures"],
                    "orientation_failures": t["orientation_failures"],
                }
            )
        gaps.append(
            {
                "file": path.name,
                "ema_fraction_closed": d["gap"]["fraction_closed"],
                "raw_fraction_closed": d["raw_diagnostic"]["gap"]["fraction_closed"]
                if "gap" in d["raw_diagnostic"]
                else None,
                "ema_retained_initial_fraction": d["gap"]["ema_retained_initial_fraction"],
            }
        )
    keys = [
        "distinct_class_fraction",
        "max_flag_file_share_played",
        "max_flag_file_share_symmetrized",
        "max_mirror_asymmetry_z",
        "mean_sequence_information_nats",
        "reflected_fraction",
    ]
    summary = {k: {"min": min(s[k] for s in samples), "max": max(s[k] for s in samples)} for k in keys}
    summary["samples"] = len(samples)
    summary["immediately_terminal_total"] = sum(s["immediately_terminal_count"] for s in samples)
    summary["legality_failures_total"] = sum(s["legality_failures"] for s in samples)
    summary["orientation_failures_total"] = sum(s["orientation_failures"] for s in samples)
    info_raw = [
        (a["mean_sequence_information_nats"], b["mean_sequence_information_nats"])
        for a, b in zip(
            [s for s in samples if s["block"] == "raw_initial"],
            [s for s in samples if s["block"] == "raw_final"],
        )
    ]
    summary["raw_final_over_initial_information_ratio"] = {
        "min": min(b / a for a, b in info_raw),
        "max": max(b / a for a, b in info_raw),
    }
    return {"summary": summary, "samples": samples, "gaps": gaps}


# --------------------------------------------------------------------------------------
# T3/T4  Cluster-aware effective units
# --------------------------------------------------------------------------------------
def var_factor(rho_b: float, rho_w: float, design: str) -> float:
    """Var(pooled estimator) = V / B * factor.  Model: Y_a = v[base,type] + w[case,type] + e[arm,case]
    with 2 tau^2 = rho_b V, 2 omega^2 = rho_w V, 2 eps^2 = (1 - rho_b - rho_w) V; type effects shared
    across seeds (worst case); three seeds pooled; 20 cases per base."""
    rest = 1.0 - rho_b - rho_w
    if design == "per_seed_library":   # also P1 (per-seed init arms)
        return rho_b + rho_w / CASES_PER_BASE + rest / (SEEDS * CASES_PER_BASE)
    if design == "shared_library":
        return rho_b + rho_w / CASES_PER_BASE + rest * (1.0 / (SEEDS * 2 * CASES_PER_BASE) + 1.0 / (2 * CASES_PER_BASE))
    if design == "single_seed":
        return rho_b + rho_w / CASES_PER_BASE + rest / CASES_PER_BASE
    raise ValueError(design)


def n_eff(B: int, rho_b: float, rho_w: float, design: str) -> float:
    return B / var_factor(rho_b, rho_w, design)


def se_of(B: int, rho_b: float, rho_w: float, design: str) -> float:
    return SD_PAIRED / math.sqrt(n_eff(B, rho_b, rho_w, design))


def d80(se: float) -> float:
    """Smallest true effect passing 'lower bound > 0 and point >= 0.05' with 80% power."""
    return max(MARGIN + Z80 * se, (Z975 + Z80) * se)


def pass_probability(d: float, se: float) -> float:
    return 1.0 - ncdf(max(Z975 - d / se, (MARGIN - d) / se))


B_GRID = [80, 112, 160, 240, 320, 480, 640]
RHO_B_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]
RHO_W_GRID = [0.0, 0.10, 0.30]


def table_units() -> dict:
    out = {"per_effective_unit": [], "designs": {}, "rho_w_sensitivity": [], "smallest_B": {}}
    for n in (256, 512, 913, 1024, 2048, 4096, 8192):
        se = SD_PAIRED / math.sqrt(n)
        out["per_effective_unit"].append(
            {"n_eff": n, "se": se, "half_width": Z975 * se, "mde80_lower_gt_0": (Z975 + Z80) * se,
             "d80_combined_rule": d80(se), "pass_prob_at_0.05": pass_probability(0.05, se),
             "pass_prob_at_0.07": pass_probability(0.07, se), "pass_prob_at_0.10": pass_probability(0.10, se)}
        )
    for design in ("per_seed_library", "shared_library", "single_seed"):
        rows = []
        for B in B_GRID:
            row = {"B": B, "cases_per_arm": CASES_PER_BASE * B}
            for rho_b in RHO_B_GRID:
                ne = n_eff(B, rho_b, 0.10, design)
                se = SD_PAIRED / math.sqrt(ne)
                row[f"rho_b={rho_b}"] = {"n_eff": ne, "se": se, "d80": d80(se)}
            rows.append(row)
        out["designs"][design] = rows
    for B in (80, 160):
        for rho_b in (0.05, 0.10):
            for rho_w in RHO_W_GRID:
                ne = n_eff(B, rho_b, rho_w, "per_seed_library")
                out["rho_w_sensitivity"].append({"B": B, "rho_b": rho_b, "rho_w": rho_w, "n_eff": ne,
                                                 "se": SD_PAIRED / math.sqrt(ne), "d80": d80(SD_PAIRED / math.sqrt(ne))})
    for target in (0.06, 0.07):
        for rho_b in RHO_B_GRID:
            for design in ("per_seed_library", "shared_library"):
                Bmin = None
                for B in range(FAMILIES, 641, FAMILIES):   # equal count per family
                    if d80(se_of(B, rho_b, 0.10, design)) <= target:
                        Bmin = B
                        break
                out["smallest_B"][f"d80<={target}|rho_b={rho_b}|{design}"] = Bmin
    return out


# --------------------------------------------------------------------------------------
# T5  Monte Carlo check of the variance model and of the base-cluster bootstrap
# --------------------------------------------------------------------------------------
def simulate_dataset(rng, B, rho_b, rho_w):
    tau = math.sqrt(rho_b * V / 2.0)
    omega = math.sqrt(rho_w * V / 2.0)
    eps = math.sqrt((1.0 - rho_b - rho_w) * V / 2.0)
    v = rng.normal(0.0, tau, size=(B, 3))                       # types E, I, L; shared across seeds
    w = rng.normal(0.0, omega, size=(B, CASES_PER_BASE, 3))
    e = rng.normal(0.0, eps, size=(B, CASES_PER_BASE, 3, SEEDS)) # arms E_k, I_k, L_k
    Y = v[:, None, :, None] + w[:, :, :, None] + e               # (B, cases, type, seed)
    D2 = Y[:, :, 0, :] - Y[:, :, 2, :]                           # P2 with per-seed library arms
    D2s = Y[:, :, 0, :] - Y[:, :, 2, :1]                         # P2 with the shared library arm (seed-1 draw)
    D1 = Y[:, :, 0, :] - Y[:, :, 1, :]                           # P1
    return D1, D2, D2s


def cluster_bootstrap_se(rng, D, reps=400):
    B = D.shape[0]
    per_base = D.reshape(B, -1).mean(axis=1)
    idx = rng.integers(0, B, size=(reps, B))
    means = per_base[idx].mean(axis=1)
    return means.std(ddof=1), np.percentile(means, 2.5), np.percentile(means, 97.5)


def table_monte_carlo() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    out = []
    for B, rho_b, rho_w in ((160, 0.05, 0.10), (80, 0.10, 0.10), (160, 0.0, 0.0)):
        R = 3000
        p1 = np.empty(R); p2 = np.empty(R); p2s = np.empty(R)
        for r in range(R):
            D1, D2, D2s = simulate_dataset(rng, B, rho_b, rho_w)
            p1[r] = D1.mean(); p2[r] = D2.mean(); p2s[r] = D2s.mean()
        boot_se = []; covered = 0; n_boot = 300
        for r in range(n_boot):
            D1, D2, D2s = simulate_dataset(rng, B, rho_b, rho_w)
            se_b, lo, hi = cluster_bootstrap_se(rng, D2)
            boot_se.append(se_b)
            covered += int(lo <= 0.0 <= hi)
        out.append(
            {"B": B, "rho_b": rho_b, "rho_w": rho_w, "replications": R,
             "P1": {"sim_se": p1.std(ddof=1), "analytic_se": se_of(B, rho_b, rho_w, "per_seed_library")},
             "P2_per_seed_library": {"sim_se": p2.std(ddof=1), "analytic_se": se_of(B, rho_b, rho_w, "per_seed_library")},
             "P2_shared_library": {"sim_se": p2s.std(ddof=1), "analytic_se": se_of(B, rho_b, rho_w, "shared_library")},
             "naive_se_if_60B_independent": SD_PAIRED / math.sqrt(60 * B),
             "cluster_bootstrap": {"datasets": n_boot, "resamples": 400, "mean_bootstrap_se": float(np.mean(boot_se)),
                                    "coverage_95": covered / n_boot}}
        )
    return out


# --------------------------------------------------------------------------------------
# T6  EMA parameter aging (parameters only; no play-strength inference)
# --------------------------------------------------------------------------------------
def table_ema_aging() -> dict:
    rows = []
    for U in (64, 128, 256, 500, 1000, 2000, 3000):
        retained = EMA_DECAY ** U
        w_last_64 = sum((1 - EMA_DECAY) * EMA_DECAY ** (U - k) for k in range(max(1, U - 63), U + 1))
        rows.append({"U": U, "retained_initial_fraction": retained, "weight_on_raw_trajectory": 1 - retained,
                     "weight_on_last_64_updates": w_last_64})
    return {"rows": rows, "decay": EMA_DECAY, "time_constant_updates": 1.0 / (1.0 - EMA_DECAY)}


# --------------------------------------------------------------------------------------
# T7  Calibration-stage and later-budget cost (games, CPU, wall, storage)
# --------------------------------------------------------------------------------------
def table_cost() -> dict:
    slots = 2560
    plies_per_period = 202
    completions = slots * plies_per_period / MEAN_PLIES_TRAINING_RULES
    rates = (30.0, 60.0, 100.0)
    def hours(games, rate): return games / rate / 3600.0
    budgets = []
    for U in (256, 1000, 2000, 3000):
        games = completions * U
        budgets.append({"periods": U, "games_per_seed": games, "cpu_hours_per_seed": games * SECONDS_PER_TEACHER_GAME / 3600.0,
                        **{f"wall_h_at_{int(r)}gps_per_seed": hours(games, r) for r in rates},
                        **{f"wall_h_at_{int(r)}gps_three_seeds": 3 * hours(games, r) for r in rates},
                        "outcome_receipts_MB_at_80B": games * 2 * 80 / 1e6,
                        "game_rows_MB_at_300B": games * 300 / 1e6,
                        "checkpoints_every_32_MB": (U // 32) * 12.9})
    screening = {}
    for label, arms in (("shared_library_16_arms", 16), ("per_seed_library_18_arms", 18)):
        games = arms * CASES_PER_BASE * 160
        screening[label] = {"arms": arms, "cases_per_arm": CASES_PER_BASE * 160, "games": games,
                            "hours_at_18.9_gps": games / 18.9 / 3600.0, "hours_at_1.5_gps": games / 1.5 / 3600.0}
    probe = {"checkpoints": 8, "games_per_model_per_checkpoint": 512, "models": 2, "games": 8 * 512 * 2,
             "wall_minutes_at_60gps": 8 * 512 * 2 / 60.0 / 60.0}
    confirmation = []
    for B in B_GRID:
        for label, arms in (("7_arms_shared_library", 7), ("9_arms_per_seed_library", 9)):
            games = arms * CASES_PER_BASE * B
            confirmation.append({"B": B, "design": label, "cases_per_arm": CASES_PER_BASE * B, "games": games,
                                 "hours_at_18.9_gps": games / 18.9 / 3600.0, "hours_at_1.5_gps": games / 1.5 / 3600.0,
                                 "paired_games_per_opponent_pooled": 3 * COLOURS * B,
                                 "paired_games_per_opponent_colour_pooled": 3 * B,
                                 "paired_games_per_family_pooled": 3 * CASES_PER_BASE * B // FAMILIES})
    return {"collector": {"slots": slots, "plies_per_period": plies_per_period, "mean_plies": MEAN_PLIES_TRAINING_RULES,
                          "expected_completions_per_period": completions, "expected_outcomes_per_period": 2 * completions,
                          "expected_outcomes_per_setup": 2 * completions / POOL,
                          "cpu_seconds_per_period": completions * SECONDS_PER_TEACHER_GAME,
                          **{f"wall_s_per_period_at_{int(r)}gps": completions / r for r in rates}},
            "budgets": budgets, "screening": screening, "probe": probe, "confirmation": confirmation}


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render(tables: dict) -> str:
    L = []
    lib = tables["library"]
    L.append("## T1 Library concentration references (setup_library_v1, 8,000 bases; canonical rank 0 = back rank, 3 = front row)\n")
    L.append("| statistic | library overall | family min (family) | family max (family) |")
    L.append("|---|---|---|---|")
    for k in STAT_KEYS:
        r = lib["family_range"][k]
        L.append(f"| {k} | {fmt(lib['overall'][k])} | {fmt(r['min'])} ({r['argmin']}) | {fmt(r['max'])} ({r['argmax']}) |")
    g = tables["g2_pools"]["summary"]
    L.append(f"\n## T2 Fresh-model baselines from the tracked G2 result files ({g['samples']} generation samples of 4,096 setups)\n")
    L.append("| statistic | min | max |")
    L.append("|---|---|---|")
    for k in ("distinct_class_fraction", "max_flag_file_share_played", "max_flag_file_share_symmetrized",
              "max_mirror_asymmetry_z", "mean_sequence_information_nats", "reflected_fraction"):
        L.append(f"| {k} | {fmt(g[k]['min'])} | {fmt(g[k]['max'])} |")
    L.append(f"| raw final / initial sequence-information ratio | {fmt(g['raw_final_over_initial_information_ratio']['min'])} | {fmt(g['raw_final_over_initial_information_ratio']['max'])} |")
    L.append(f"\nimmediately terminal {g['immediately_terminal_total']}, legality failures {g['legality_failures_total']}, orientation failures {g['orientation_failures_total']} over all samples.\n")
    L.append("| G2 result file | EMA fraction of gap closed | raw fraction of gap closed | EMA retained initial-parameter fraction |")
    L.append("|---|---|---|---|")
    for gp in tables["g2_pools"]["gaps"]:
        L.append(f"| {gp['file']} | {fmt(gp['ema_fraction_closed'])} | {fmt(gp['raw_fraction_closed'])} | {fmt(gp['ema_retained_initial_fraction'])} |")
    u = tables["units"]
    L.append(f"\n## T3 Planning constants\n\nSD_paired {SD_PAIRED}, V {fmt(V)}, single-lane SD {SD_SINGLE_LANE}, implied same-case cross-arm correlation {fmt(RHO_CASE)}, tie fraction {TIE_FRACTION}, margin {MARGIN}, alpha two-sided 0.05, power 0.80.\n")
    L.append("## T4a Resolution per effective independent paired unit\n")
    L.append("| n_eff | SE | 95% half-width | MDE (lower>0, 80%) | d80 combined rule | P(pass at 0.05) | P(pass at 0.07) | P(pass at 0.10) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in u["per_effective_unit"]:
        L.append(f"| {r['n_eff']} | {fmt(r['se'])} | {fmt(r['half_width'])} | {fmt(r['mde80_lower_gt_0'])} | {fmt(r['d80_combined_rule'])} | {fmt(r['pass_prob_at_0.05'],3)} | {fmt(r['pass_prob_at_0.07'],3)} | {fmt(r['pass_prob_at_0.10'],3)} |")
    for design, title in (("per_seed_library", "T4b P2 with per-seed library arms (also P1): n_eff / SE / d80, rho_w = 0.10"),
                          ("shared_library", "T4c P2 with one shared library arm: n_eff / SE / d80, rho_w = 0.10"),
                          ("single_seed", "T4d one seed alone: n_eff / SE / d80, rho_w = 0.10")):
        L.append(f"\n## {title}\n")
        L.append("| B bases | cases/arm | " + " | ".join(f"rho_b={r}" for r in RHO_B_GRID) + " |")
        L.append("|---|---|" + "---|" * len(RHO_B_GRID))
        for row in u["designs"][design]:
            cells = [f"{row[f'rho_b={r}']['n_eff']:.0f} / {row[f'rho_b={r}']['se']:.4f} / {row[f'rho_b={r}']['d80']:.4f}" for r in RHO_B_GRID]
            L.append(f"| {row['B']} | {row['cases_per_arm']} | " + " | ".join(cells) + " |")
    L.append("\n## T4e rho_w sensitivity (per-seed library arms)\n")
    L.append("| B | rho_b | rho_w | n_eff | SE | d80 |")
    L.append("|---|---|---|---|---|---|")
    for r in u["rho_w_sensitivity"]:
        L.append(f"| {r['B']} | {r['rho_b']} | {r['rho_w']} | {r['n_eff']:.0f} | {fmt(r['se'])} | {fmt(r['d80'])} |")
    L.append("\n## T4f Smallest B (equal per family) with d80 at or below the target, rho_w = 0.10\n")
    L.append("| target d80 | design | " + " | ".join(f"rho_b={r}" for r in RHO_B_GRID) + " |")
    L.append("|---|---|" + "---|" * len(RHO_B_GRID))
    for target in (0.06, 0.07):
        for design in ("per_seed_library", "shared_library"):
            cells = [str(u["smallest_B"][f"d80<={target}|rho_b={r}|{design}"]) for r in RHO_B_GRID]
            L.append(f"| {target} | {design} | " + " | ".join(cells) + " |")
    L.append("\n## T5 Monte Carlo check of the variance model and the base-cluster bootstrap\n")
    L.append("| B | rho_b | rho_w | P1 SE sim / analytic | P2 per-seed-L SE sim / analytic | P2 shared-L SE sim / analytic | naive SE (60B indep.) | bootstrap mean SE | 95% coverage |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in tables["monte_carlo"]:
        L.append(f"| {r['B']} | {r['rho_b']} | {r['rho_w']} | {fmt(r['P1']['sim_se'])} / {fmt(r['P1']['analytic_se'])} | {fmt(r['P2_per_seed_library']['sim_se'])} / {fmt(r['P2_per_seed_library']['analytic_se'])} | {fmt(r['P2_shared_library']['sim_se'])} / {fmt(r['P2_shared_library']['analytic_se'])} | {fmt(r['naive_se_if_60B_independent'])} | {fmt(r['cluster_bootstrap']['mean_bootstrap_se'])} | {fmt(r['cluster_bootstrap']['coverage_95'],3)} |")
    e = tables["ema_aging"]
    L.append(f"\n## T6 EMA parameter aging (decay {e['decay']}, time constant {e['time_constant_updates']:.0f} updates) — parameters only\n")
    L.append("| U | retained initial-parameter fraction 0.999^U | weight on the raw trajectory | weight on the last 64 updates |")
    L.append("|---|---|---|---|")
    for r in e["rows"]:
        L.append(f"| {r['U']} | {fmt(r['retained_initial_fraction'])} | {fmt(r['weight_on_raw_trajectory'])} | {fmt(r['weight_on_last_64_updates'])} |")
    c = tables["cost"]
    k = c["collector"]
    L.append(f"\n## T7a Asynchronous collector per period: {k['slots']} slots x {k['plies_per_period']} plies / {k['mean_plies']} mean plies\n")
    L.append(f"expected completions per period {k['expected_completions_per_period']:.0f}, outcomes {k['expected_outcomes_per_period']:.0f}, mean outcomes per setup {k['expected_outcomes_per_setup']:.2f}, CPU s per period {k['cpu_seconds_per_period']:.0f}, wall s per period at 30/60/100 games/s: {k['wall_s_per_period_at_30gps']:.0f} / {k['wall_s_per_period_at_60gps']:.0f} / {k['wall_s_per_period_at_100gps']:.0f}\n")
    L.append("## T7b Collection cost per seed by period budget (games are cost only; no play-strength inference)\n")
    L.append("| periods | games/seed | CPU h/seed | wall h/seed @30 | @60 | @100 | three seeds @30 | @60 | @100 | receipts MB | game rows MB | checkpoints MB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in c["budgets"]:
        L.append(f"| {r['periods']} | {r['games_per_seed']/1e6:.2f} M | {r['cpu_hours_per_seed']:.1f} | {r['wall_h_at_30gps_per_seed']:.1f} | {r['wall_h_at_60gps_per_seed']:.1f} | {r['wall_h_at_100gps_per_seed']:.1f} | {r['wall_h_at_30gps_three_seeds']:.1f} | {r['wall_h_at_60gps_three_seeds']:.1f} | {r['wall_h_at_100gps_three_seeds']:.1f} | {r['outcome_receipts_MB_at_80B']:.0f} | {r['game_rows_MB_at_300B']:.0f} | {r['checkpoints_every_32_MB']:.0f} |")
    L.append("\n## T7c Screening-slice evaluation accounting (160 bases x 20 cases = 3,200 cases per arm)\n")
    L.append("| design | arms | cases/arm | games | hours @18.9 games/s | hours @1.5 games/s |")
    L.append("|---|---|---|---|---|---|")
    for label, r in c["screening"].items():
        L.append(f"| {label} | {r['arms']} | {r['cases_per_arm']} | {r['games']} | {r['hours_at_18.9_gps']:.2f} | {r['hours_at_1.5_gps']:.2f} |")
    p = c["probe"]
    L.append(f"\nTeacher-regime learning-curve probe (optional): {p['checkpoints']} checkpoints x {p['models']} models x {p['games_per_model_per_checkpoint']} games = {p['games']} teacher games, about {p['wall_minutes_at_60gps']:.0f} minutes at 60 games/s.\n")
    L.append("## T7d Confirmation-slice accounting by B (not opened in the calibration stage)\n")
    L.append("| B | design | cases/arm | games | hours @18.9 | hours @1.5 | paired games per opponent (pooled) | per opponent x colour | per family |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in c["confirmation"]:
        L.append(f"| {r['B']} | {r['design']} | {r['cases_per_arm']} | {r['games']} | {r['hours_at_18.9_gps']:.2f} | {r['hours_at_1.5_gps']:.2f} | {r['paired_games_per_opponent_pooled']} | {r['paired_games_per_opponent_colour_pooled']} | {r['paired_games_per_family_pooled']} |")
    return "\n".join(L) + "\n"


def build() -> dict:
    return {"library": table_library(), "g2_pools": table_g2_pools(), "units": table_units(),
            "monte_carlo": table_monte_carlo(), "ema_aging": table_ema_aging(), "cost": table_cost(),
            "constants": {"SD_PAIRED": SD_PAIRED, "SD_SINGLE_LANE": SD_SINGLE_LANE, "V": V, "RHO_CASE": RHO_CASE,
                          "SECONDS_PER_TEACHER_GAME": SECONDS_PER_TEACHER_GAME, "MEAN_PLIES_TRAINING_RULES": MEAN_PLIES_TRAINING_RULES,
                          "RNG_SEED": RNG_SEED}}


def canonical(tables: dict) -> str:
    return json.dumps(tables, indent=1, sort_keys=True, allow_nan=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    tables = build()
    text = canonical(tables)
    if not a.quiet:
        print(render(tables))
    digest = hashlib.sha256(text.encode()).hexdigest()
    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
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
