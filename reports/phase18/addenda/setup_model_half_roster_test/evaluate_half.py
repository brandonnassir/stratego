#!/usr/bin/env python3
"""Evaluate setup-model checkpoints as formation sources for the 8 handcrafted bots.
Same protocol as the first addendum: each bot plays the frozen 2,560-game G3 schedule with
its own formation from (a checkpoint's EMA model | the init model | a same-family library
formation), paired by case, G3 bootstrap unchanged."""
import importlib.util, json, sys, time, dataclasses
from pathlib import Path
from types import SimpleNamespace
import torch
torch.set_num_threads(2)
ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
SRC = ROOT / "output/phase18/worktrees/g3-stage6b"; sys.path.insert(0, str(SRC))
OUT = ROOT / "output/phase18/runtime/addendum_half_roster" / (sys.argv[sys.argv.index("--half") + 1] if "--half" in sys.argv else "A")
from stratego.training.phase18 import g3_evaluation as ev
from stratego.training.phase18.g3_contract import HANDCRAFTED_OPPONENTS
from stratego.training.phase18.setup_contract import SetupTrainingConfig
from stratego.training.phase18.setup_learning import SetupTrainer
from stratego.training.phase18.setup_model import state_dict_digest
from stratego.evaluation.registry import policy_ref
from stratego.evaluation.match_spec import MatchSpec
from stratego.evaluation.match_runner import run_schedule

HALVES = {
    "A": ("basic_heuristic", "tactical_rule_based", "stress_miner_rush", "stress_information_miser"),
    "B": ("strategic_rule_based", "stress_scout_rush", "stress_berserker", "stress_chaos"),
}
HALF = sys.argv[sys.argv.index("--half") + 1] if "--half" in sys.argv else "A"
assert HALF in HALVES
spec = importlib.util.spec_from_file_location("driver", SRC / "scripts/phase18_g3_pilot.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
NAMESPACE = driver.NAMESPACE          # the G3 evaluation seeds: same schedule, same case seeds
NS = "phase18_addendum_library_setup_from_scratch_v1"; RUN_ID = "ADDENDUM-HALF-ROSTER-" + (sys.argv[sys.argv.index("--half") + 1] if "--half" in sys.argv else "A") + "-2026"; WORKERS = 8

def log(m):
    line = f"[eval {time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(OUT / "eval.log", "a") as f: f.write(line + "\n")

def main():
    args = [x for x in sys.argv[1:] if x not in ("--half", HALF)]; ckpts = [int(x) for x in args] or [0, 256, 384, 512, 640, 768, 896, 1024]
    bases = ev.load_evaluation_bases(); cases = ev.build_cases(bases); by_key = {(e.family_id, e.base_index): e for e in bases}
    config = SetupTrainingConfig(run_id=RUN_ID, device="cpu", pool_size=1024)
    arms, digests = {}, {}
    for c in ckpts:
        tr, _ = SetupTrainer.load_checkpoint(OUT / f"ckpt_{c:04d}", config, namespace=NS, seed_index=1, device="cpu")
        model = tr.evaluation_model(device="cpu"); digests[f"ckpt_{c}"] = state_dict_digest(model)
        gen = ev.resolve_own_setups(model, cases, namespace=NAMESPACE, seed_index=1, device="cpu")
        arms[f"ckpt_{c}"] = (ev.build_arm_bank(cases, gen.samples), [s.content_fingerprint for s in gen.samples])
    pairs, fps = [], []
    for case in cases:
        own = by_key[(case.family_id, 400 + ((case.base_index - 400 + 5) % 10))]
        own_e = ev.validate_setup(ev.orient_setup(tuple(int(v) for v in own.canonical_setup), case.colour), case.colour)
        oc = ev.RED if case.colour == ev.BLUE else ev.BLUE
        base_e = ev.validate_setup(ev.orient_setup(case.base_canonical, oc), oc)
        red, blue = (own_e, base_e) if case.colour == ev.RED else (base_e, own_e)
        pairs.append(ev.SetupPair(setup_pair_id=int(case.case_index), red_setup=red, blue_setup=blue, generation_seed=0,
                                  bank_version=ev.G3_EVALUATION_BANK_VERSION, generation_family="addendum_library_same_family_rot5"))
        fps.append(own.base_setup_id)
    arms["library"] = (ev.SetupBank(bank_version=ev.G3_EVALUATION_BANK_VERSION, root_seed=0, generation_family="addendum_library_same_family_rot5", pairs=tuple(pairs)), fps)
    log(f"arms: {list(arms)} | digests {dict((k, v[:10]) for k, v in digests.items())}")
    root = ev.evaluation_schedule_seed(NAMESPACE)
    rows = {}
    for student in HANDCRAFTED_OPPONENTS:
        for arm, (bank, _) in arms.items():
            path = OUT / "eval" / f"rows_{student}_{arm}.jsonl"; path.parent.mkdir(exist_ok=True)
            if path.exists():
                rows[(student, arm)] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]; continue
            ms = [MatchSpec(candidate=policy_ref(student), opponent=policy_ref(c.opponent_id), setup_pair_id=int(c.case_index),
                            candidate_color=int(c.colour), replicate=0, root_seed=root, setup_bank_version=ev.G3_EVALUATION_BANK_VERSION,
                            rules=ev.PLAY_EVALUATION_RULES) for c in cases]
            for m in ms: m.resolve_setups(bank)
            t = time.perf_counter(); run = run_schedule(ms, bank, worker_count=WORKERS, record_actions=False)
            out = [{"match_id": r.match_id, "setup_pair_id": r.setup_pair_id, "candidate_score": r.candidate_score, "errored": bool(r.policy_error)} for r in run.results]
            path.write_text("".join(json.dumps(o) + "\n" for o in out)); rows[(student, arm)] = out
            log(f"{student:26s} {arm:9s} EWR {sum(o['candidate_score'] for o in out)/len(out):.4f} errors {run.policy_errors} {time.perf_counter()-t:.0f}s")
    def ns(rd, off=0): return [SimpleNamespace(match_id=o["match_id"], setup_pair_id=o["setup_pair_id"] + off, candidate_score=o["candidate_score"], errored=o["errored"]) for o in rd]
    pooled_cases = [dataclasses.replace(c, case_index=s * 2560 + c.case_index) for s in range(len(HANDCRAFTED_OPPONENTS)) for c in cases]
    def pooled(a, b):
        ra, rb = [], []
        for s, st in enumerate(HANDCRAFTED_OPPONENTS): ra += ns(rows[(st, a)], s * 2560); rb += ns(rows[(st, b)], s * 2560)
        r = ev.paired_analysis(ra, rb, pooled_cases, namespace=NAMESPACE)
        return {k: r[k] for k in ("point", "lower", "upper", "candidate_ewr", "control_ewr", "cases_on_which_arms_differ", "bootstrap_standard_error", "by_opponent")}
    analysis = {"digests": digests, "ewr": {st: {arm: sum(o["candidate_score"] for o in rows[(st, arm)]) / 2560 for arm in arms} for st in HANDCRAFTED_OPPONENTS}, "pooled": {}, "per_student": {}}
    model_arms = [f"ckpt_{c}" for c in ckpts]
    contrasts = [(a, "ckpt_0") for a in model_arms if a != "ckpt_0"] + [(a, "library") for a in model_arms] + [("ckpt_0", "library")]
    for a, b in dict.fromkeys(contrasts): analysis["pooled"][f"{a}-{b}"] = pooled(a, b)
    final = model_arms[-1]
    for st in HANDCRAFTED_OPPONENTS:
        analysis["per_student"][st] = {}
        for a, b in ((final, "ckpt_0"), (final, "library")):
            r = ev.paired_analysis(ns(rows[(st, a)]), ns(rows[(st, b)]), cases, namespace=NAMESPACE)
            analysis["per_student"][st][f"{a}-{b}"] = {k: r[k] for k in ("point", "lower", "upper")}
    seen = set(HALVES[HALF]); analysis["half"] = HALF; analysis["seen_opponents"] = sorted(seen)
    for key in list(analysis["pooled"]):
        bo = analysis["pooled"][key]["by_opponent"]
        analysis["pooled"][key]["seen_opponents_mean"] = sum(bo[o]["mean_difference"] for o in bo if o in seen) / max(1, sum(1 for o in bo if o in seen))
        analysis["pooled"][key]["unseen_opponents_mean"] = sum(bo[o]["mean_difference"] for o in bo if o not in seen) / max(1, sum(1 for o in bo if o not in seen))
    analysis["ewr_by_student_group"] = {arm: {"seen_students": sum(analysis["ewr"][s][arm] for s in seen) / len(seen), "unseen_students": sum(analysis["ewr"][s][arm] for s in analysis["ewr"] if s not in seen) / (len(analysis["ewr"]) - len(seen))} for arm in arms}
    (OUT / "eval_analysis.json").write_text(json.dumps(analysis, indent=1, sort_keys=True) + "\n"); log("EVAL DONE")

if __name__ == "__main__":
    main()
