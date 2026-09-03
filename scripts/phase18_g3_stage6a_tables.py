#!/usr/bin/env python3
"""Phase 18 Agent 6, Stage 6A: every numerical table in the G3 analysis, reproducibly.

Corrective revision (Stage 6A, second commit). Reads tracked evidence only:

* data/setups/setup_library_v1.jsonl and its manifest    structural references, TRAIN split only
* reports/phase18/phase18_evaluation_contract_v1.json    instrument SD, tie fraction, margin
* stratego/training/phase18/setup_contract.py             EMA decay, pool, batch, epochs
* stratego/engine/constants.py                            TRAINING_RULES / EVALUATION_RULES
* reports/phase_8_data/agent_02_corpus_manifest.json      mean plies under battleless 100
* reports/phase_4_data/agent_04_baseline_league_raw.csv   seconds per rule-vs-rule game
* reports/phase18/g1_random_confirmation/run_v1.json      G1 harness games per second
* reports/phase17/local_eval/results/*.result.json        Phase 17 in-process harness games per second
* reports/phase18/phase18_g1_random_confirmation_contract_v1.json   C0 checkpoint bytes
* reports/phase18/g2/... and reports/phase18/g2_raw_confirmation/... seed result files
                                                          fresh-model generation samples, EMA/raw gap

Data boundary: the library is one file, so all 8,000 lines are read from disk; a non-train
line has only its split tag matched and is never decoded, and only train-split rows enter any
statistic. The first Stage 6A commit (286acb33) decoded every row, including the reserved
validation bases 410..449 and the test bases 450..499; that structural exposure (piece positions,
never an outcome, game or performance number) is recorded in the JSON and is not undone. No Stratego game is played, no pool sampled, no model built.

Usage
-----
    python scripts/phase18_g3_stage6a_tables.py --write   # print tables, write the JSON
    python scripts/phase18_g3_stage6a_tables.py --check   # recompute and compare with the JSON

Monte Carlo sections use numpy's PCG64 with fixed seeds, so the JSON is byte-stable.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/phase18/g3_design/phase18_g3_stage6a_tables_v1.json"

Z975 = 1.959963984540054
Z80 = 0.8416212335729143
Z90 = 1.2815515655446004
OPPONENTS = 10
COLOURS = 2
CASES_PER_BASE = OPPONENTS * COLOURS
SEEDS = 3
FAMILIES = 16
RNG_SEED = 20260902

# Frozen reviewer decisions (2026-09-02)
POWERED_EFFECT = 0.06
MIN_B = 160                      # 10 bases per family
MAX_B = 640                      # the reserved validation bases 410..449, 40 per family
STRATUM_FLOOR = 200              # evaluation contract: worst-stratum regression needs >= 200 paired games
UPPER_BOUND_ONE_SIDED = 0.95     # predeclared conservative bound on the screening-derived per-base variance

# Provisional collector parameters (T fixed; S provisional until preflight)
SLOTS_PROVISIONAL = 2560
PLIES_PER_PERIOD = 202           # published cadence: 2 x train_every_per_player (method map S21)
TARGET_OUTCOMES_PER_SETUP = 4.0

# Constants that have no tracked source and are therefore declared here (each is cited in the analysis)
OUTCOME_RECEIPT_BYTES = 80       # artifacts/phase18/g2_raw_confirmation_v1/seed_1/outcome_receipts.jsonl (untracked): 20.9 MB / 262,144
CHECKPOINT_MB = 12.9             # raw.pt + optimizer.pt + ema.pt of the G2 raw confirmation (untracked)
GAME_ROW_BYTES = 300             # planning assumption for a compact collection-game row


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def chi2_ppf_wilson_hilferty(p: float, nu: float) -> float:
    z = {0.05: -1.6448536269514722, 0.95: 1.6448536269514722}[p]
    return nu * (1.0 - 2.0 / (9.0 * nu) + z * math.sqrt(2.0 / (9.0 * nu))) ** 3


# --------------------------------------------------------------------------------------
# Frozen constants from tracked sources
# --------------------------------------------------------------------------------------
def load_constants() -> dict:
    src = {}
    contract = json.load(open(ROOT / "reports/phase18/phase18_evaluation_contract_v1.json"))
    plan = contract["power_and_precision_plan"]
    own = plan["measured_inputs"]["own_setup_instrument"]
    src["SD_PAIRED"] = (own["sd_of_paired_per_board_difference"], "phase18_evaluation_contract_v1.json:power_and_precision_plan.measured_inputs.own_setup_instrument.sd_of_paired_per_board_difference")
    src["TIE_FRACTION"] = (own["tie_fraction"], "phase18_evaluation_contract_v1.json:...own_setup_instrument.tie_fraction")
    src["SD_SINGLE_LANE"] = (plan["measured_inputs"]["single_lane_sd"], "phase18_evaluation_contract_v1.json:...measured_inputs.single_lane_sd")
    src["MARGIN"] = (plan["predeclared_practical_margins"]["value"], "phase18_evaluation_contract_v1.json:...predeclared_practical_margins.value")

    text = (ROOT / "stratego/training/phase18/setup_contract.py").read_text()
    def const(name):
        m = re.search(rf"^{name}\s*=\s*([0-9.e_-]+)", text, re.M)
        if not m:
            raise RuntimeError(f"{name} not found in setup_contract.py")
        return float(m.group(1).replace("_", ""))
    src["EMA_DECAY"] = (const("SETUP_EMA_DECAY"), "stratego/training/phase18/setup_contract.py:SETUP_EMA_DECAY")
    src["POOL_SIZE"] = (int(const("SETUP_POOL_SIZE")), "stratego/training/phase18/setup_contract.py:SETUP_POOL_SIZE")
    src["BATCH_SIZE"] = (int(const("SETUP_BATCH_SIZE")), "stratego/training/phase18/setup_contract.py:SETUP_BATCH_SIZE")
    src["EPOCHS_PER_UPDATE"] = (int(const("SETUP_EPOCHS_PER_UPDATE")), "stratego/training/phase18/setup_contract.py:SETUP_EPOCHS_PER_UPDATE")

    eng = (ROOT / "stratego/engine/constants.py").read_text()
    def rules(name):
        m = re.search(rf"^{name}\s*=\s*RulesConfig\(\s*battleless_move_limit=(\d+),\s*absolute_move_limit=(\d+)", eng, re.M | re.S)
        if not m:
            raise RuntimeError(f"{name} not found in engine/constants.py")
        return int(m.group(1)), int(m.group(2))
    tb, ta = rules("TRAINING_RULES")
    eb, ea = rules("EVALUATION_RULES")
    src["TRAINING_BATTLELESS"] = (tb, "stratego/engine/constants.py:TRAINING_RULES")
    src["TRAINING_ABSOLUTE"] = (ta, "stratego/engine/constants.py:TRAINING_RULES")
    src["EVALUATION_BATTLELESS"] = (eb, "stratego/engine/constants.py:EVALUATION_RULES")
    src["EVALUATION_ABSOLUTE"] = (ea, "stratego/engine/constants.py:EVALUATION_RULES")

    manifest = json.load(open(ROOT / "reports/phase_8_data/agent_02_corpus_manifest.json"))
    src["MEAN_PLIES_TRAINING_RULES"] = (manifest["decision_totals"]["per_split"]["train"]["mean_plies"], "reports/phase_8_data/agent_02_corpus_manifest.json:decision_totals.per_split.train.mean_plies")

    secs = []
    with open(ROOT / "reports/phase_4_data/agent_04_baseline_league_raw.csv") as f:
        for row in csv.DictReader(f):
            secs.append(float(row["wall_clock_seconds"]))
    src["SECONDS_PER_TEACHER_GAME"] = (sum(secs) / len(secs), f"reports/phase_4_data/agent_04_baseline_league_raw.csv: mean wall_clock_seconds over {len(secs)} rule-vs-rule games")
    src["TEACHER_GAMES_MEASURED"] = (len(secs), "same file, row count")

    run = json.load(open(ROOT / "reports/phase18/g1_random_confirmation/run_v1.json"))
    games = sum(run["arms"][a]["completed"] for a in ("candidate", "reference"))
    src["G1_HARNESS_GAMES_PER_SECOND"] = (games / run["seconds"], "reports/phase18/g1_random_confirmation/run_v1.json: completed games / seconds")

    walls, played = 0.0, 0
    for path in sorted(glob.glob(str(ROOT / "reports/phase17/local_eval/results/*.result.json"))):
        d = json.load(open(path))
        a = datetime.fromisoformat(d["started_utc"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(d["finished_utc"].replace("Z", "+00:00"))
        walls += (b - a).total_seconds()
        played += sum(v["games"] for v in d["lane_results"].values())
    src["PHASE17_HARNESS_GAMES_PER_SECOND"] = (played / walls, "reports/phase17/local_eval/results/*.result.json: games / (finished_utc - started_utc), 25 candidates")

    g1c = json.load(open(ROOT / "reports/phase18/phase18_g1_random_confirmation_contract_v1.json"))
    src["C0_CHECKPOINT_BYTES"] = (g1c["checkpoints"]["candidate"]["bytes"], "reports/phase18/phase18_g1_random_confirmation_contract_v1.json:checkpoints.candidate.bytes")
    src["C0_CHECKPOINT_SHA256"] = (g1c["checkpoints"]["candidate"]["sha256"], "same file, checkpoints.candidate.sha256")

    lib_manifest = json.load(open(ROOT / "data/setups/setup_library_v1_manifest.json"))
    src["LIBRARY_SPLIT_COUNTS"] = (lib_manifest["split_counts"], "data/setups/setup_library_v1_manifest.json:split_counts")
    src["LIBRARY_SPLIT_RULE"] = (lib_manifest["split_rule"], "data/setups/setup_library_v1_manifest.json:split_rule")
    src["LIBRARY_DIGEST"] = (lib_manifest["library_content_digest"], "data/setups/setup_library_v1_manifest.json:library_content_digest")
    return src


C = load_constants()
SD_PAIRED = C["SD_PAIRED"][0]
SD_SINGLE_LANE = C["SD_SINGLE_LANE"][0]
V = SD_PAIRED ** 2
SIGMA2 = SD_SINGLE_LANE ** 2
RHO_CASE = 1.0 - V / (2.0 * SIGMA2)
MARGIN = C["MARGIN"][0]
EMA_DECAY = C["EMA_DECAY"][0]
POOL = C["POOL_SIZE"][0]
MEAN_PLIES = C["MEAN_PLIES_TRAINING_RULES"][0]
SEC_PER_GAME = C["SECONDS_PER_TEACHER_GAME"][0]
RETENTION_PERIODS = math.ceil(C["TRAINING_ABSOLUTE"][0] / PLIES_PER_PERIOD) + 1
SE_TARGET_POINT = (POWERED_EFFECT - MARGIN) / Z80          # point >= margin with 80% power at d = 0.06
SE_TARGET_LOWER = POWERED_EFFECT / (Z975 + Z80)             # lower bound > 0 with 80% power at d = 0.06
SE_TARGET = min(SE_TARGET_POINT, SE_TARGET_LOWER)


# --------------------------------------------------------------------------------------
# T1  Library structural references, TRAIN split only
# --------------------------------------------------------------------------------------
STAT_KEYS = [
    "front_row_flag_share", "back_rank_flag_share", "max_flag_file_share_symmetrized",
    "bombs_adjacent_to_flag_mean", "front_row_bomb_share", "back_rank_bomb_share",
    "max_bomb_file_share_symmetrized",
]


def setup_stats(setups: list[str]) -> dict:
    n = len(setups)
    flag_rank, flag_file, bomb_file = Counter(), Counter(), Counter()
    adj, front_bombs, back_bombs = [], [], []
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
    return {
        "n": n,
        "front_row_flag_share": flag_rank[3] / n,
        "back_rank_flag_share": flag_rank[0] / n,
        "max_flag_file_share_symmetrized": max((flag_file[f] + flag_file[9 - f]) / (2.0 * n) for f in range(5)),
        "bombs_adjacent_to_flag_mean": sum(adj) / n,
        "front_row_bomb_share": sum(front_bombs) / (10.0 * n),
        "back_rank_bomb_share": sum(back_bombs) / (10.0 * n),
        "max_bomb_file_share_symmetrized": max((bomb_file[f] + bomb_file[9 - f]) / (2.0 * 6 * n) for f in range(5)),
    }


def table_library() -> dict:
    parsed = 0
    by_split = Counter()
    by_family: dict[str, list[str]] = {}
    split_tag = re.compile(r'"split":\s*"([a-z]+)"')
    with open(ROOT / "data/setups/setup_library_v1.jsonl") as f:
        for line in f:
            # The library is one file, so every line is read from disk; only the split tag of a
            # non-train line is matched, and only train lines are decoded into setup content.
            parsed += 1
            tag = split_tag.search(line)
            split = tag.group(1) if tag else "unknown"
            by_split[split] += 1
            if split == "train":
                r = json.loads(line)
                by_family.setdefault(r["family_key"], []).append(r["canonical_setup"])
    train_setups = [s for v in by_family.values() for s in v]
    overall = setup_stats(train_setups)
    fam = {k: setup_stats(v) for k, v in sorted(by_family.items())}
    family_range = {
        key: {"min": min(fam[k][key] for k in fam), "max": max(fam[k][key] for k in fam),
              "argmin": min(fam, key=lambda k: fam[k][key]), "argmax": max(fam, key=lambda k: fam[k][key])}
        for key in STAT_KEYS
    }
    boundary = {
        "library_lines_scanned": parsed,
        "lines_by_split_tag": dict(by_split),
        "rows_decoded_for_references": len(train_setups),
        "splits_decoded": ["train"],
        "exposure": "this run decodes the structural content (piece positions) of train-split rows only; "
                    "validation and test lines are read from disk as part of the single file but only their "
                    "split tag is matched; no outcome, game or performance number of any base is read or produced",
        "stage_6a_first_commit_note": "the Stage 6A commit 286acb33 computed references over all 8,000 rows, "
                                      "including the reserved validation bases 410..449 and the test bases 450..499; "
                                      "that structural exposure is recorded as feature exposure and is not undone",
    }
    return {"overall_train": overall, "per_family_train": fam, "family_range_train": family_range, "data_boundary": boundary}


# --------------------------------------------------------------------------------------
# T2  Fresh-model baselines from the tracked G2 result files
# --------------------------------------------------------------------------------------
def table_g2_pools() -> dict:
    files = [ROOT / f"reports/phase18/g2/phase18_g2_seed_{s}_result_v1.json" for s in (1, 2, 3)]
    files += [ROOT / f"reports/phase18/g2_raw_confirmation/phase18_g2_raw_confirmation_seed_{s}_result_v1.json" for s in (1, 2, 3)]
    samples, gaps = [], []
    for path in files:
        d = json.load(open(path))
        blocks = {"ema_initial": d["initial"]["generation_telemetry"], "ema_final": d["final"]["generation_telemetry"],
                  "raw_initial": d["raw_diagnostic"]["initial"]["generation_telemetry"],
                  "raw_final": d["raw_diagnostic"]["final"]["generation_telemetry"]}
        for label, t in blocks.items():
            played = t["flag_file_histogram_played"]
            n = t["count"]
            samples.append({
                "file": path.name, "block": label, "count": n,
                "distinct_class_fraction": t["distinct_class_fingerprints"] / n,
                "max_flag_file_share_played": max(played) / n,
                "max_flag_file_share_symmetrized": max((played[f] + played[9 - f]) / (2.0 * n) for f in range(5)),
                "max_mirror_asymmetry_z": max(abs(played[f] - played[9 - f]) / math.sqrt(max(1, played[f] + played[9 - f])) for f in range(5)),
                "mean_sequence_information_nats": t["mean_sequence_information_nats"],
                "reflected_fraction": t["reflected_fraction"],
                "immediately_terminal_count": t["immediately_terminal_count"],
                "legality_failures": t["legality_failures"], "orientation_failures": t["orientation_failures"]})
        gaps.append({"file": path.name, "ema_fraction_closed": d["gap"]["fraction_closed"],
                     "raw_fraction_closed": d["raw_diagnostic"]["gap"]["fraction_closed"],
                     "ema_retained_initial_fraction": d["gap"]["ema_retained_initial_fraction"]})
    keys = ["distinct_class_fraction", "max_flag_file_share_played", "max_flag_file_share_symmetrized",
            "max_mirror_asymmetry_z", "mean_sequence_information_nats", "reflected_fraction"]
    summary = {k: {"min": min(s[k] for s in samples), "max": max(s[k] for s in samples)} for k in keys}
    summary["samples"] = len(samples)
    summary["immediately_terminal_total"] = sum(s["immediately_terminal_count"] for s in samples)
    summary["legality_failures_total"] = sum(s["legality_failures"] for s in samples)
    summary["orientation_failures_total"] = sum(s["orientation_failures"] for s in samples)
    ini = [s for s in samples if s["block"] == "raw_initial"]
    fin = [s for s in samples if s["block"] == "raw_final"]
    ratios = [b["mean_sequence_information_nats"] / a["mean_sequence_information_nats"] for a, b in zip(ini, fin)]
    summary["raw_final_over_initial_information_ratio"] = {"min": min(ratios), "max": max(ratios)}
    return {"summary": summary, "samples": samples, "gaps": gaps}


# --------------------------------------------------------------------------------------
# Variance model (three seeds, per-seed library arms = the frozen design)
# --------------------------------------------------------------------------------------
def var_factor(rho_b: float, rho_w: float, design: str) -> float:
    rest = 1.0 - rho_b - rho_w
    if design == "per_seed_library":
        return rho_b + rho_w / CASES_PER_BASE + rest / (SEEDS * CASES_PER_BASE)
    if design == "shared_library":
        return rho_b + rho_w / CASES_PER_BASE + rest * (1.0 / (SEEDS * 2 * CASES_PER_BASE) + 1.0 / (2 * CASES_PER_BASE))
    if design == "single_seed":
        return rho_b + rho_w / CASES_PER_BASE + rest / CASES_PER_BASE
    raise ValueError(design)


def per_base_variance(rho_b: float, rho_w: float, design: str = "per_seed_library") -> float:
    """Variance of one base's mean paired difference (all cases, all seeds) = V * factor."""
    return V * var_factor(rho_b, rho_w, design)


def se_of(B: int, rho_b: float, rho_w: float, design: str = "per_seed_library") -> float:
    return math.sqrt(per_base_variance(rho_b, rho_w, design) / B)


def n_eff(B: int, rho_b: float, rho_w: float, design: str = "per_seed_library") -> float:
    return B / var_factor(rho_b, rho_w, design)


def d80(se: float) -> float:
    return max(MARGIN + Z80 * se, (Z975 + Z80) * se)


def pass_probability(d: float, se: float) -> float:
    return 1.0 - ncdf(max(Z975 - d / se, (MARGIN - d) / se))


def ceil16(x: float) -> int:
    return int(16 * math.ceil(x / 16.0))


def sizing_rule(s2_base: float, df: int) -> dict:
    """Frozen rule: B = smallest multiple of 16 >= MIN_B with sqrt(s2_upper / B) <= SE_TARGET, else REVIEW."""
    factor = df / chi2_ppf_wilson_hilferty(0.05, df)
    s2_upper = s2_base * factor
    b_raw = s2_upper / SE_TARGET ** 2
    b = max(MIN_B, ceil16(b_raw))
    return {"s2_base": s2_base, "df": df, "upper_bound_factor": factor, "s2_upper": s2_upper,
            "B_unrounded": b_raw, "B": b if b <= MAX_B else None, "decision": "size" if b <= MAX_B else "REVIEW: no valid size within the 640 reserved bases"}


B_GRID = [160, 208, 256, 320, 400, 480, 560, 640]
RHO_B_GRID = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]


def table_units() -> dict:
    out = {"per_effective_unit": [], "frozen_design": [], "single_seed": [], "shared_library_comparison": [],
           "rho_w_sensitivity": [], "sizing_rule": [], "sizing_reference_points": [],
           "se_target": {"point_criterion": SE_TARGET_POINT, "lower_bound_criterion": SE_TARGET_LOWER, "binding": SE_TARGET}}
    for n in (256, 512, 913, 1024, 2048, 4096, 8192):
        se = SD_PAIRED / math.sqrt(n)
        out["per_effective_unit"].append({"n_eff": n, "se": se, "half_width": Z975 * se, "mde80_lower_gt_0": (Z975 + Z80) * se,
                                          "d80_combined_rule": d80(se), "pass_prob_at_0.05": pass_probability(0.05, se),
                                          "pass_prob_at_0.06": pass_probability(0.06, se), "pass_prob_at_0.08": pass_probability(0.08, se)})
    for B in B_GRID:
        row = {"B": B, "cases_per_arm": CASES_PER_BASE * B, "games_9_arms": 9 * CASES_PER_BASE * B}
        for rho_b in RHO_B_GRID:
            se = se_of(B, rho_b, 0.10)
            row[f"rho_b={rho_b}"] = {"n_eff": n_eff(B, rho_b, 0.10), "se": se, "d80": d80(se), "per_base_sd": math.sqrt(per_base_variance(rho_b, 0.10))}
        out["frozen_design"].append(row)
        row1 = {"B": B}
        for rho_b in RHO_B_GRID:
            se = se_of(B, rho_b, 0.10, "single_seed")
            row1[f"rho_b={rho_b}"] = {"n_eff": n_eff(B, rho_b, 0.10, "single_seed"), "se": se}
        out["single_seed"].append(row1)
    for rho_b in RHO_B_GRID:
        out["shared_library_comparison"].append({"B": 160, "rho_b": rho_b, "rho_w": 0.10,
                                                 "n_eff_per_seed_library": n_eff(160, rho_b, 0.10), "games_9_arms": 9 * 3200,
                                                 "n_eff_shared_library": n_eff(160, rho_b, 0.10, "shared_library"), "games_7_arms": 7 * 3200})
    for B in (160, 256):
        for rho_b in (0.05, 0.10):
            for rho_w in (0.0, 0.10, 0.30):
                se = se_of(B, rho_b, rho_w)
                out["rho_w_sensitivity"].append({"B": B, "rho_b": rho_b, "rho_w": rho_w, "n_eff": n_eff(B, rho_b, rho_w), "se": se, "d80": d80(se)})
    df = MIN_B - FAMILIES
    for s_base in (0.07, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30):
        out["sizing_rule"].append({"per_base_sd_measured": s_base, **sizing_rule(s_base ** 2, df)})
    for rho_b, rho_w in ((0.0, 0.0), (0.02, 0.10), (0.05, 0.10), (0.10, 0.10), (0.20, 0.10), (0.30, 0.10)):
        s2 = per_base_variance(rho_b, rho_w)
        out["sizing_reference_points"].append({"rho_b": rho_b, "rho_w": rho_w, "per_base_sd_model": math.sqrt(s2), **sizing_rule(s2, df)})
    return out


# --------------------------------------------------------------------------------------
# Monte Carlo machinery
# --------------------------------------------------------------------------------------
def make_family_effect(rho_f: float, seed: int) -> np.ndarray:
    """A FIXED family x type effect (types E, I, L), centered over families so the pooled estimand is unchanged."""
    rng = np.random.default_rng(seed)
    F = rng.normal(0.0, math.sqrt(rho_f * V / 2.0), size=(FAMILIES, 3))
    return F - F.mean(axis=0, keepdims=True)


def simulate(rng, B, rho_b, rho_w, F, mu_E, mu_I, mu_L):
    """Per-seed case differences D1[k,b,c] = E_k - I_k and D2[k,b,c] = E_k - L_k under the model.
    mu_E: array (SEEDS,) of per-seed trained means; mu_I, mu_L scalars."""
    tau = math.sqrt(rho_b * V / 2.0)
    omega = math.sqrt(rho_w * V / 2.0)
    eps = math.sqrt((1.0 - rho_b - rho_w) * V / 2.0)
    nf = B // FAMILIES
    fam = np.repeat(np.arange(FAMILIES), nf)                          # family of each base
    v = rng.normal(0.0, tau, size=(B, 3))
    w = rng.normal(0.0, omega, size=(B, CASES_PER_BASE, 3))
    e = rng.normal(0.0, eps, size=(3, SEEDS, B, CASES_PER_BASE))       # (type, seed, base, case)
    base_type = F[fam] + v                                             # (B, 3)
    Y = base_type.T[:, None, :, None] + w.transpose(2, 0, 1)[:, None, :, :] + e   # (type, seed, base, case)
    mu = np.zeros((3, SEEDS))
    mu[0] = mu_E
    mu[1] = mu_I
    mu[2] = mu_L
    Y = Y + mu[:, :, None, None]
    D1 = Y[0] - Y[1]
    D2 = Y[0] - Y[2]
    return D1, D2, fam


def stratified_bootstrap(rng, per_base: np.ndarray, reps: int):
    """Resample bases with replacement WITHIN each family (equal bases per family), carrying the
    base's full per-base mean (all opponents, colours, arms and seeds already averaged into it).

    Finite-stratum rescaling: resampling n_f bases within a stratum reproduces (n_f - 1) / n_f of the
    stratum variance, so the resample means are rescaled about their centre by sqrt(n_f / (n_f - 1))
    (1.054 at 10 bases per family, 1.033 at 16). The percentile interval and the SE are read from the
    rescaled distribution. The uncorrected version undercovers (about 0.92 at 10 per family)."""
    B = per_base.size
    nf = B // FAMILIES
    idx = rng.integers(0, nf, size=(reps, FAMILIES, nf)) + (np.arange(FAMILIES) * nf)[None, :, None]
    means = per_base[idx.reshape(reps, -1)].mean(axis=1)
    centre = per_base.mean()
    return centre + (means - centre) * math.sqrt(nf / (nf - 1.0))


def global_bootstrap(rng, per_base: np.ndarray, reps: int):
    B = per_base.size
    idx = rng.integers(0, B, size=(reps, B))
    return per_base[idx].mean(axis=1)


def direct_per_base_variance(per_base: np.ndarray) -> float:
    B = per_base.size
    nf = B // FAMILIES
    groups = per_base.reshape(FAMILIES, nf)
    return float(groups.var(axis=1, ddof=1).mean())


def rho_decomposition(D: np.ndarray, fam: np.ndarray) -> tuple[float, float]:
    """Method-of-moments decomposition after removing fixed family means.
    m1 = within-seed, within-base, across-case covariance -> 2 tau^2 (= rho_b V)
    m2 = across-seed, same-case covariance               -> 2 tau^2 + 2 omega^2 (= (rho_b + rho_w) V)
    Raw cross-seed correlation therefore estimates rho_b + rho_w, never rho_w alone."""
    K, B, Cc = D.shape
    Dc = D.copy()
    for f in range(FAMILIES):
        sel = fam == f
        Dc[:, sel, :] -= Dc[:, sel, :].mean()
    Vhat = float((Dc ** 2).mean())
    s_c = Dc.sum(axis=2)                                   # (K, B)
    m1 = float(((s_c ** 2 - (Dc ** 2).sum(axis=2)) / (Cc * (Cc - 1))).mean())
    s_k = Dc.sum(axis=0)                                   # (B, C)
    m2 = float(((s_k ** 2 - (Dc ** 2).sum(axis=0)) / (K * (K - 1))).mean())
    return m1 / Vhat, (m2 - m1) / Vhat


def table_monte_carlo() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    F = make_family_effect(0.05, RNG_SEED + 1)
    out = []
    for B, rho_b, rho_w in ((160, 0.05, 0.10), (256, 0.10, 0.10), (160, 0.0, 0.0)):
        R = 3000
        p1 = np.empty(R); p2 = np.empty(R); s1 = np.empty(R)
        for r in range(R):
            D1, D2, fam = simulate(rng, B, rho_b, rho_w, F, np.zeros(SEEDS), 0.0, 0.0)
            p1[r] = D1.mean(); p2[r] = D2.mean(); s1[r] = D2[0].mean()
        nboot = 300
        strat_se, glob_se, s2_direct, s2_upper_ok, cov_strat, cov_glob = [], [], [], 0, 0, 0
        rb, rw = [], []
        true_s2 = per_base_variance(rho_b, rho_w)
        df = B - FAMILIES
        factor = df / chi2_ppf_wilson_hilferty(0.05, df)
        for r in range(nboot):
            D1, D2, fam = simulate(rng, B, rho_b, rho_w, F, np.zeros(SEEDS), 0.0, 0.0)
            per_base = D2.mean(axis=(0, 2))
            ms = stratified_bootstrap(rng, per_base, 400)
            mg = global_bootstrap(rng, per_base, 400)
            strat_se.append(ms.std(ddof=1)); glob_se.append(mg.std(ddof=1))
            cov_strat += int(np.percentile(ms, 2.5) <= 0.0 <= np.percentile(ms, 97.5))
            cov_glob += int(np.percentile(mg, 2.5) <= 0.0 <= np.percentile(mg, 97.5))
            s2 = direct_per_base_variance(per_base)
            s2_direct.append(s2)
            s2_upper_ok += int(s2 * factor >= true_s2)
            b_hat, w_hat = rho_decomposition(D2, fam)
            rb.append(b_hat); rw.append(w_hat)
        out.append({
            "B": B, "rho_b": rho_b, "rho_w": rho_w, "family_effect_fraction": 0.05, "replications": R,
            "P1": {"sim_se": p1.std(ddof=1), "analytic_se": se_of(B, rho_b, rho_w)},
            "P2": {"sim_se": p2.std(ddof=1), "analytic_se": se_of(B, rho_b, rho_w)},
            "single_seed_P2": {"sim_se": s1.std(ddof=1), "analytic_se": se_of(B, rho_b, rho_w, "single_seed")},
            "naive_se_if_60B_independent": SD_PAIRED / math.sqrt(60 * B),
            "stratified_bootstrap": {"datasets": nboot, "resamples": 400, "finite_stratum_rescaling": math.sqrt((B // FAMILIES) / (B // FAMILIES - 1.0)), "mean_se": float(np.mean(strat_se)), "coverage_95": cov_strat / nboot},
            "global_bootstrap": {"mean_se": float(np.mean(glob_se)), "coverage_95": cov_glob / nboot},
            "direct_per_base_variance": {"true": true_s2, "mean_estimate": float(np.mean(s2_direct)),
                                         "se_from_mean_estimate": math.sqrt(float(np.mean(s2_direct)) / B),
                                         "upper_bound_factor": factor, "upper_bound_coverage": s2_upper_ok / nboot},
            "rho_decomposition": {"rho_b_mean": float(np.mean(rb)), "rho_b_sd": float(np.std(rb, ddof=1)),
                                  "rho_w_mean": float(np.mean(rw)), "rho_w_sd": float(np.std(rw, ddof=1)),
                                  "raw_cross_seed_correlation_estimates": rho_b + rho_w}})
    return out


def table_full_gate() -> dict:
    """Probability of passing the complete G3 statistical gate, conditional on three realized seeds whose
    true effects are (d - delta, d, d + delta)."""
    rng = np.random.default_rng(RNG_SEED + 7)
    F = make_family_effect(0.05, RNG_SEED + 1)
    rows = []
    R = 1000
    nboot = 300
    for rho_b, rho_w in ((0.05, 0.10), (0.10, 0.10)):
        for B in (160, 256):
            for d1, d2 in ((0.13, 0.05), (0.14, 0.06), (0.16, 0.08), (0.18, 0.10)):
                for delta in (0.0, 0.02, 0.04):
                    mu_E = np.array([d1 - delta, d1, d1 + delta])
                    mu_I, mu_L = 0.0, d1 - d2
                    n_p1 = n_p2 = n_dir = n_all = 0
                    for r in range(R):
                        D1, D2, fam = simulate(rng, B, rho_b, rho_w, F, mu_E, mu_I, mu_L)
                        pb1 = D1.mean(axis=(0, 2)); pb2 = D2.mean(axis=(0, 2))
                        point1, point2 = pb1.mean(), pb2.mean()
                        lb1 = np.percentile(stratified_bootstrap(rng, pb1, nboot), 2.5)
                        lb2 = np.percentile(stratified_bootstrap(rng, pb2, nboot), 2.5)
                        pass1 = (lb1 > 0.0) and (point1 >= MARGIN)
                        pass2 = (lb2 > 0.0) and (point2 >= MARGIN)
                        direction = bool(np.all(D1.mean(axis=(1, 2)) > 0.0) and np.all(D2.mean(axis=(1, 2)) > 0.0))
                        n_p1 += pass1; n_p2 += pass2; n_dir += direction; n_all += (pass1 and pass2 and direction)
                    rows.append({"rho_b": rho_b, "rho_w": rho_w, "B": B, "true_P1": d1, "true_P2": d2, "seed_spread": delta,
                                 "replications": R, "P1_power": n_p1 / R, "P2_power": n_p2 / R,
                                 "direction_all_seeds": n_dir / R, "full_gate_power": n_all / R,
                                 "analytic_P2_pass_prob": pass_probability(d2, se_of(B, rho_b, rho_w))})
    return {"rows": rows, "rule": "P1 and P2 each: stratified-bootstrap 2.5th percentile > 0 and point >= 0.05; "
                                  "direction: every realized seed's P1 and P2 point estimates > 0; full gate = all of them",
            "conditioning": "true per-seed effects are fixed at (d - delta, d, d + delta): inference is conditional on the three realized training seeds"}


def table_ema_aging() -> dict:
    rows = []
    for U in (64, 128, 256, 500, 1000, 2000, 3000):
        retained = EMA_DECAY ** U
        w_last_64 = sum((1 - EMA_DECAY) * EMA_DECAY ** (U - k) for k in range(max(1, U - 63), U + 1))
        rows.append({"U": U, "retained_initial_fraction": retained, "weight_on_raw_trajectory": 1 - retained, "weight_on_last_64_updates": w_last_64})
    return {"rows": rows, "decay": EMA_DECAY, "time_constant_updates": 1.0 / (1.0 - EMA_DECAY)}


def table_cost() -> dict:
    completions = SLOTS_PROVISIONAL * PLIES_PER_PERIOD / MEAN_PLIES
    slots_for_target = TARGET_OUTCOMES_PER_SETUP * POOL / 2.0 * MEAN_PLIES / PLIES_PER_PERIOD
    rates = (30.0, 60.0, 100.0)
    def hours(games, rate): return games / rate / 3600.0
    budgets = []
    for U in (256, 1000, 2000, 3000):
        games = completions * U
        budgets.append({"periods": U, "games_per_seed": games, "cpu_hours_per_seed": games * SEC_PER_GAME / 3600.0,
                        **{f"wall_h_at_{int(r)}gps_per_seed": hours(games, r) for r in rates},
                        **{f"wall_h_at_{int(r)}gps_three_seeds": 3 * hours(games, r) for r in rates},
                        "outcome_receipts_MB": games * 2 * OUTCOME_RECEIPT_BYTES / 1e6,
                        "game_rows_MB": games * GAME_ROW_BYTES / 1e6,
                        "checkpoints_every_32_MB": (U // 32) * CHECKPOINT_MB})
    g17 = C["PHASE17_HARNESS_GAMES_PER_SECOND"][0]
    g1 = C["G1_HARNESS_GAMES_PER_SECOND"][0]
    screening_games = 18 * CASES_PER_BASE * MIN_B
    probe_games = 8 * 2 * 512
    screening = {"arms": 18, "cases_per_arm": CASES_PER_BASE * MIN_B, "games": screening_games,
                 "hours_in_process": screening_games / g17 / 3600.0, "hours_g1_harness": screening_games / g1 / 3600.0,
                 "probe_games": probe_games, "probe_minutes_at_60gps": probe_games / 60.0 / 60.0}
    confirmation = []
    for B in B_GRID:
        games = 9 * CASES_PER_BASE * B
        confirmation.append({"B": B, "cases_per_arm": CASES_PER_BASE * B, "games": games,
                             "hours_in_process": games / g17 / 3600.0, "hours_g1_harness": games / g1 / 3600.0,
                             "paired_games_per_opponent_pooled": 3 * COLOURS * B, "paired_games_per_opponent_colour_pooled": 3 * B,
                             "paired_games_per_family_pooled": 3 * CASES_PER_BASE * B // FAMILIES,
                             "paired_games_per_opponent_single_seed": COLOURS * B})
    return {"collector": {"slots_provisional": SLOTS_PROVISIONAL, "plies_per_period": PLIES_PER_PERIOD, "mean_plies": MEAN_PLIES,
                          "retention_periods": RETENTION_PERIODS, "expected_completions_per_period": completions,
                          "expected_outcomes_per_period": 2 * completions, "expected_outcomes_per_setup": 2 * completions / POOL,
                          "slots_giving_exactly_four_outcomes_per_setup": slots_for_target,
                          "cpu_seconds_per_period": completions * SEC_PER_GAME,
                          **{f"wall_s_per_period_at_{int(r)}gps": completions / r for r in rates}},
            "budgets": budgets, "screening": screening, "confirmation": confirmation,
            "harness_rates": {"in_process_games_per_second": g17, "g1_games_per_second": g1}}


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render(t: dict) -> str:
    L = []
    lib = t["library"]
    b = lib["data_boundary"]
    L.append("## T1 Library structural references, TRAIN split only (setup_library_v1 base_index 0..399; canonical rank 0 = back rank, 3 = front row)\n")
    L.append(f"Lines scanned {b['library_lines_scanned']} (split tags {b['lines_by_split_tag']}); rows decoded and used {b['rows_decoded_for_references']} (train only).\n")
    L.append("| statistic | train overall | family min (family) | family max (family) |")
    L.append("|---|---|---|---|")
    for k in STAT_KEYS:
        r = lib["family_range_train"][k]
        L.append(f"| {k} | {fmt(lib['overall_train'][k])} | {fmt(r['min'])} ({r['argmin']}) | {fmt(r['max'])} ({r['argmax']}) |")
    g = t["g2_pools"]["summary"]
    L.append(f"\n## T2 Fresh-model baselines from the tracked G2 result files ({g['samples']} generation samples of 4,096 setups)\n")
    L.append("| statistic | min | max |")
    L.append("|---|---|---|")
    for k in ("distinct_class_fraction", "max_flag_file_share_played", "max_flag_file_share_symmetrized", "max_mirror_asymmetry_z", "mean_sequence_information_nats", "reflected_fraction"):
        L.append(f"| {k} | {fmt(g[k]['min'])} | {fmt(g[k]['max'])} |")
    rr = g["raw_final_over_initial_information_ratio"]
    L.append(f"| raw final / initial sequence-information ratio | {fmt(rr['min'])} | {fmt(rr['max'])} |")
    L.append(f"\nimmediately terminal {g['immediately_terminal_total']}, legality failures {g['legality_failures_total']}, orientation failures {g['orientation_failures_total']} over all samples.\n")
    L.append("| G2 result file | EMA fraction of gap closed | raw fraction of gap closed | EMA retained initial-parameter fraction |")
    L.append("|---|---|---|---|")
    for gp in t["g2_pools"]["gaps"]:
        L.append(f"| {gp['file']} | {fmt(gp['ema_fraction_closed'])} | {fmt(gp['raw_fraction_closed'])} | {fmt(gp['ema_retained_initial_fraction'])} |")
    L.append("\n## T3 Frozen constants and their tracked sources\n")
    L.append("| constant | value | source |")
    L.append("|---|---|---|")
    for k, (v, s) in t["constants"].items():
        L.append(f"| {k} | {fmt(v, 4) if isinstance(v, float) else v} | {s} |")
    L.append(f"\nDerived: V = SD_paired^2 = {fmt(V)}; implied same-case cross-arm correlation {fmt(RHO_CASE)}; SE target for d80 <= {POWERED_EFFECT}: point criterion {fmt(SE_TARGET_POINT)}, lower-bound criterion {fmt(SE_TARGET_LOWER)}, binding {fmt(SE_TARGET)}; retention {RETENTION_PERIODS} periods = ceil(absolute limit / T) + 1.\n")
    u = t["units"]
    L.append("## T4a Resolution per effective independent paired unit\n")
    L.append("| n_eff | SE | 95% half-width | MDE (lower>0, 80%) | d80 combined rule | P(pass at 0.05) | P(pass at 0.06) | P(pass at 0.08) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in u["per_effective_unit"]:
        L.append(f"| {r['n_eff']} | {fmt(r['se'])} | {fmt(r['half_width'])} | {fmt(r['mde80_lower_gt_0'])} | {fmt(r['d80_combined_rule'])} | {fmt(r['pass_prob_at_0.05'],3)} | {fmt(r['pass_prob_at_0.06'],3)} | {fmt(r['pass_prob_at_0.08'],3)} |")
    L.append("\n## T4b Frozen design (three seeds, per-seed library arms; P1 and P2): n_eff / SE / d80, rho_w = 0.10\n")
    L.append("| B bases | cases/arm | games (9 arms) | " + " | ".join(f"rho_b={r}" for r in RHO_B_GRID) + " |")
    L.append("|---|---|---|" + "---|" * len(RHO_B_GRID))
    for row in u["frozen_design"]:
        cells = [f"{row[f'rho_b={r}']['n_eff']:.0f} / {row[f'rho_b={r}']['se']:.4f} / {row[f'rho_b={r}']['d80']:.4f}" for r in RHO_B_GRID]
        L.append(f"| {row['B']} | {row['cases_per_arm']} | {row['games_9_arms']} | " + " | ".join(cells) + " |")
    L.append("\n## T4c One seed alone: n_eff / SE, rho_w = 0.10\n")
    L.append("| B bases | " + " | ".join(f"rho_b={r}" for r in RHO_B_GRID) + " |")
    L.append("|---|" + "---|" * len(RHO_B_GRID))
    for row in u["single_seed"]:
        L.append(f"| {row['B']} | " + " | ".join(f"{row[f'rho_b={r}']['n_eff']:.0f} / {row[f'rho_b={r}']['se']:.4f}" for r in RHO_B_GRID) + " |")
    L.append("\n## T4d Why per-seed library arms (B = 160, rho_w = 0.10): effective units per design\n")
    L.append("| rho_b | per-seed arms n_eff (28,800 games) | shared arm n_eff (22,400 games) |")
    L.append("|---|---|---|")
    for r in u["shared_library_comparison"]:
        L.append(f"| {r['rho_b']} | {r['n_eff_per_seed_library']:.0f} | {r['n_eff_shared_library']:.0f} |")
    L.append("\n## T4e rho_w sensitivity (frozen design)\n")
    L.append("| B | rho_b | rho_w | n_eff | SE | d80 |")
    L.append("|---|---|---|---|---|---|")
    for r in u["rho_w_sensitivity"]:
        L.append(f"| {r['B']} | {r['rho_b']} | {r['rho_w']} | {r['n_eff']:.0f} | {fmt(r['se'])} | {fmt(r['d80'])} |")
    L.append(f"\n## T4f Frozen sizing rule: measured per-base SD on the screening slice -> confirmation B (df = {MIN_B - FAMILIES}, one-sided 95% upper bound on the variance, SE target {fmt(SE_TARGET)})\n")
    L.append("| measured per-base SD | s2 | upper-bound factor | s2_upper | B unrounded | B (multiple of 16, >= 160) | decision |")
    L.append("|---|---|---|---|---|---|---|")
    for r in u["sizing_rule"]:
        L.append(f"| {fmt(r['per_base_sd_measured'],3)} | {fmt(r['s2_base'])} | {fmt(r['upper_bound_factor'],3)} | {fmt(r['s2_upper'])} | {r['B_unrounded']:.1f} | {r['B'] if r['B'] else '-'} | {r['decision']} |")
    L.append("\n## T4g Sizing-rule reference points implied by the variance model (for orientation; the rule uses the measured value)\n")
    L.append("| rho_b | rho_w | model per-base SD | s2_upper | B unrounded | B | decision |")
    L.append("|---|---|---|---|---|---|---|")
    for r in u["sizing_reference_points"]:
        L.append(f"| {r['rho_b']} | {r['rho_w']} | {fmt(r['per_base_sd_model'])} | {fmt(r['s2_upper'])} | {r['B_unrounded']:.1f} | {r['B'] if r['B'] else '-'} | {r['decision']} |")
    L.append("\n## T5 Monte Carlo: variance model, stratified cluster bootstrap, direct per-base variance, rho decomposition (fixed family effect 0.05 V)\n")
    L.append("| B | rho_b | rho_w | P1 SE sim / analytic | P2 SE sim / analytic | single-seed SE sim / analytic | naive SE (60B) | stratified boot SE / coverage | global boot SE / coverage | direct s2 mean / true | upper-bound factor / coverage | rho_b est mean +- sd | rho_w est mean +- sd |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in t["monte_carlo"]:
        sb, gb, dv, rd = r["stratified_bootstrap"], r["global_bootstrap"], r["direct_per_base_variance"], r["rho_decomposition"]
        L.append(f"| {r['B']} | {r['rho_b']} | {r['rho_w']} | {fmt(r['P1']['sim_se'])} / {fmt(r['P1']['analytic_se'])} | {fmt(r['P2']['sim_se'])} / {fmt(r['P2']['analytic_se'])} | {fmt(r['single_seed_P2']['sim_se'])} / {fmt(r['single_seed_P2']['analytic_se'])} | {fmt(r['naive_se_if_60B_independent'])} | {fmt(sb['mean_se'])} / {fmt(sb['coverage_95'],3)} | {fmt(gb['mean_se'])} / {fmt(gb['coverage_95'],3)} | {fmt(dv['mean_estimate'])} / {fmt(dv['true'])} | {fmt(dv['upper_bound_factor'],3)} / {fmt(dv['upper_bound_coverage'],3)} | {fmt(rd['rho_b_mean'],3)} +- {fmt(rd['rho_b_sd'],3)} | {fmt(rd['rho_w_mean'],3)} +- {fmt(rd['rho_w_sd'],3)} |")
    e = t["ema_aging"]
    L.append(f"\n## T6 EMA parameter aging (decay {e['decay']}, time constant {e['time_constant_updates']:.0f} updates) — parameters only\n")
    L.append("| U | retained initial-parameter fraction | weight on the raw trajectory | weight on the last 64 updates |")
    L.append("|---|---|---|---|")
    for r in e["rows"]:
        L.append(f"| {r['U']} | {fmt(r['retained_initial_fraction'])} | {fmt(r['weight_on_raw_trajectory'])} | {fmt(r['weight_on_last_64_updates'])} |")
    c = t["cost"]
    k = c["collector"]
    L.append(f"\n## T7a Collector per period: S = {k['slots_provisional']} slots (provisional) x T = {k['plies_per_period']} plies / {k['mean_plies']:.1f} mean plies; retention {k['retention_periods']} periods\n")
    L.append(f"expected completions per period {k['expected_completions_per_period']:.0f}, outcomes {k['expected_outcomes_per_period']:.0f}, mean outcomes per generated setup {k['expected_outcomes_per_setup']:.2f} (S giving exactly 4.0: {k['slots_giving_exactly_four_outcomes_per_setup']:.0f}); CPU s per period {k['cpu_seconds_per_period']:.0f}; wall s per period at 30/60/100 games/s: {k['wall_s_per_period_at_30gps']:.0f} / {k['wall_s_per_period_at_60gps']:.0f} / {k['wall_s_per_period_at_100gps']:.0f}\n")
    L.append("## T7b Collection cost per seed by period budget (cost only; no play-strength inference)\n")
    L.append("| periods | games/seed | CPU h/seed | wall h/seed @30 | @60 | @100 | three seeds @30 | @60 | @100 | receipts MB | game rows MB | checkpoints MB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in c["budgets"]:
        L.append(f"| {r['periods']} | {r['games_per_seed']/1e6:.2f} M | {r['cpu_hours_per_seed']:.1f} | {r['wall_h_at_30gps_per_seed']:.1f} | {r['wall_h_at_60gps_per_seed']:.1f} | {r['wall_h_at_100gps_per_seed']:.1f} | {r['wall_h_at_30gps_three_seeds']:.1f} | {r['wall_h_at_60gps_three_seeds']:.1f} | {r['wall_h_at_100gps_three_seeds']:.1f} | {r['outcome_receipts_MB']:.0f} | {r['game_rows_MB']:.0f} | {r['checkpoints_every_32_MB']:.0f} |")
    s = c["screening"]; hr = c["harness_rates"]
    L.append(f"\n## T7c Screening-slice evaluation accounting (160 bases x 20 cases = 3,200 cases per arm; harness rates {hr['in_process_games_per_second']:.1f} and {hr['g1_games_per_second']:.2f} games/s)\n")
    L.append("| arms | cases/arm | games | hours in-process (after R1–R11) | hours G1 harness (fallback) | probe teacher games | probe minutes @60 games/s |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| {s['arms']} | {s['cases_per_arm']} | {s['games']} | {s['hours_in_process']:.2f} | {s['hours_g1_harness']:.2f} | {s['probe_games']} | {s['probe_minutes_at_60gps']:.1f} |")
    L.append("\n## T7d Confirmation-slice accounting by B (9 arms; not opened in the calibration stage)\n")
    L.append("| B | cases/arm | games | hours in-process | hours G1 harness | paired games per opponent (pooled) | per opponent x colour (pooled) | per family (pooled) | per opponent (one seed) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in c["confirmation"]:
        L.append(f"| {r['B']} | {r['cases_per_arm']} | {r['games']} | {r['hours_in_process']:.2f} | {r['hours_g1_harness']:.2f} | {r['paired_games_per_opponent_pooled']} | {r['paired_games_per_opponent_colour_pooled']} | {r['paired_games_per_family_pooled']} | {r['paired_games_per_opponent_single_seed']} |")
    fg = t["full_gate"]
    L.append("\n## T8 Monte Carlo full-gate power, conditional on three realized seeds with true effects (d - delta, d, d + delta)\n")
    L.append("| rho_b | rho_w | B | true P1 | true P2 | seed spread | P1 power | P2 power | direction in all seeds | FULL GATE | analytic P2 pass prob |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in fg["rows"]:
        L.append(f"| {r['rho_b']} | {r['rho_w']} | {r['B']} | {r['true_P1']} | {r['true_P2']} | {r['seed_spread']} | {fmt(r['P1_power'],3)} | {fmt(r['P2_power'],3)} | {fmt(r['direction_all_seeds'],3)} | {fmt(r['full_gate_power'],3)} | {fmt(r['analytic_P2_pass_prob'],3)} |")
    return "\n".join(L) + "\n"


def build() -> dict:
    return {"constants": {k: list(v) for k, v in C.items()}, "library": table_library(), "g2_pools": table_g2_pools(),
            "units": table_units(), "monte_carlo": table_monte_carlo(), "full_gate": table_full_gate(),
            "ema_aging": table_ema_aging(), "cost": table_cost(),
            "frozen_decisions": {"powered_effect": POWERED_EFFECT, "min_B": MIN_B, "max_B": MAX_B, "library_arm": "independent per training seed",
                                 "confirmation_B": "selected only at the post-calibration review by the frozen sizing rule",
                                 "confirmation_bases": "drawn per family from the reserved validation bases 410..449 by a frozen deterministic seed",
                                 "plies_per_period": PLIES_PER_PERIOD, "slots_provisional": SLOTS_PROVISIONAL,
                                 "target_outcomes_per_setup": TARGET_OUTCOMES_PER_SETUP, "rng_seed": RNG_SEED}}


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
        print(render(tables), end="")
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
