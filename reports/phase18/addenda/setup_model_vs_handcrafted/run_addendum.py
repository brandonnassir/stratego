#!/usr/bin/env python3
"""Operator addendum (informal, not gate evidence).

Question: does the learned setup model make the HANDCRAFTED bots better?
No neural policy anywhere. Each of the 8 handcrafted bots plays the frozen G3
evaluation schedule (160 library formations x 8 opponents x 2 colours = 2,560
paired games) three times, differing only in where its OWN starting formation
comes from:

  learned  - the G3 candidate's period-256 EMA setup model (trained only on
             handcrafted-teacher games; digest ea1a809b...)
  frozen   - the same model at initialisation, never trained (082ff778...)
  library  - a handcrafted library formation from the same family (rotated so
             it is never the opponent's own formation)

The learned/frozen arms use exactly the G3 evaluation seeds, so the formations
are byte-identical to the ones candidate_final / control_final played in G3 —
only the mover changes, from the neural policy to a handcrafted bot.
"""
import importlib.util, json, sys, time, dataclasses
from pathlib import Path
from types import SimpleNamespace

import torch
torch.set_num_threads(2)

ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
SRC = ROOT / "output/phase18/worktrees/g3-stage6b"
sys.path.insert(0, str(SRC))
RT = ROOT / "output/phase18/runtime/g3_pilot_v2"
OUT = ROOT / "output/phase18/runtime/addendum_setup_vs_handcrafted"
OUT.mkdir(parents=True, exist_ok=True)

from stratego.training.phase18 import g3_evaluation as ev
from stratego.training.phase18.g3_contract import HANDCRAFTED_OPPONENTS
from stratego.training.phase18.g3_bundle import load_setup_trainer
from stratego.training.phase18.setup_model import state_dict_digest
from stratego.evaluation.registry import policy_ref
from stratego.evaluation.match_spec import MatchSpec
from stratego.evaluation.match_runner import run_schedule

spec = importlib.util.spec_from_file_location("driver", SRC / "scripts/phase18_g3_pilot.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
NAMESPACE = driver.NAMESPACE
WORKERS = 8

def log(msg):
    line = f"[addendum {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a") as f: f.write(line + "\n")

def main():
    # ---- cases: the frozen G3 case set ------------------------------------------
    bases = ev.load_evaluation_bases()
    cases = ev.build_cases(bases)
    assert len(cases) == 2560
    by_key = {(e.family_id, e.base_index): e for e in bases}
    log(f"{len(cases)} cases over {len(bases)} bases")

    # ---- the three own-setup sources ----------------------------------------------
    config = driver.production_config("candidate", c1_device="mps")
    learned_tr, _ = load_setup_trainer(RT / "candidate/bundles/bundle_0256", config, device="cpu")
    frozen_tr, _ = load_setup_trainer(RT / "candidate/bundles/bundle_0000", config, device="cpu")
    learned = learned_tr.evaluation_model(device="cpu")
    frozen = frozen_tr.evaluation_model(device="cpu")
    digests = {"learned": state_dict_digest(learned), "frozen": state_dict_digest(frozen)}
    assert digests["learned"].startswith("ea1a809b") and digests["frozen"].startswith("082ff778")
    log(f"learned EMA {digests['learned'][:12]} | frozen EMA {digests['frozen'][:12]}")

    def model_bank(model):
        gen = ev.resolve_own_setups(model, cases, namespace=NAMESPACE, seed_index=1, device="cpu")
        return ev.build_arm_bank(cases, gen.samples), [s.content_fingerprint for s in gen.samples]

    def library_bank():
        pairs, fps = [], []
        for case in cases:
            own_index = 400 + ((case.base_index - 400 + 5) % 10)        # never the opponent's base
            own = by_key[(case.family_id, own_index)]
            own_engine = ev.validate_setup(ev.orient_setup(tuple(int(v) for v in own.canonical_setup), case.colour), case.colour)
            opp_colour = ev.RED if case.colour == ev.BLUE else ev.BLUE
            base_engine = ev.validate_setup(ev.orient_setup(case.base_canonical, opp_colour), opp_colour)
            red, blue = (own_engine, base_engine) if case.colour == ev.RED else (base_engine, own_engine)
            pairs.append(ev.SetupPair(setup_pair_id=int(case.case_index), red_setup=red, blue_setup=blue,
                                      generation_seed=0, bank_version=ev.G3_EVALUATION_BANK_VERSION,
                                      generation_family="addendum_library_same_family_rot5"))
            fps.append(own.base_setup_id)
        return ev.SetupBank(bank_version=ev.G3_EVALUATION_BANK_VERSION, root_seed=0,
                            generation_family="addendum_library_same_family_rot5", pairs=tuple(pairs)), fps

    banks = {}
    banks["learned"], fp_learned = model_bank(learned)
    banks["frozen"], fp_frozen = model_bank(frozen)
    banks["library"], fp_library = library_bank()
    log(f"own setups differ learned vs frozen on {sum(a != b for a, b in zip(fp_learned, fp_frozen))}/2560 cases")

    # ---- play ------------------------------------------------------------------------
    root_seed = ev.evaluation_schedule_seed(NAMESPACE)
    def schedule(student):
        ms = []
        for case in cases:
            ms.append(MatchSpec(candidate=policy_ref(student), opponent=policy_ref(case.opponent_id),
                                setup_pair_id=int(case.case_index), candidate_color=int(case.colour), replicate=0,
                                root_seed=root_seed, setup_bank_version=ev.G3_EVALUATION_BANK_VERSION,
                                rules=ev.PLAY_EVALUATION_RULES))
        return ms

    rows = {}   # (student, arm) -> list of row dicts
    summaries = {}
    for student in HANDCRAFTED_OPPONENTS:
        for arm in ("learned", "frozen", "library"):
            path = OUT / f"rows_{student}_{arm}.jsonl"
            if path.exists():
                rows[(student, arm)] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                log(f"{student:26s} {arm:8s} reused {len(rows[(student, arm)])} rows"); continue
            ms = schedule(student)
            for m in ms: m.resolve_setups(banks[arm])
            t = time.perf_counter()
            run = run_schedule(ms, banks[arm], worker_count=WORKERS, record_actions=False)
            dt = time.perf_counter() - t
            out = [{"match_id": r.match_id, "setup_pair_id": r.setup_pair_id, "candidate_score": r.candidate_score,
                    "candidate_result": r.candidate_result, "errored": bool(r.policy_error), "plies": r.plies,
                    "terminal_reason": r.terminal_reason, "red_setup": r.red_setup, "blue_setup": r.blue_setup}
                   for r in run.results]
            path.write_text("".join(json.dumps(o) + "\n" for o in out))
            rows[(student, arm)] = out
            ewr = sum(o["candidate_score"] for o in out) / len(out)
            summaries[f"{student}/{arm}"] = {"games": len(out), "ewr": ewr, "policy_errors": run.policy_errors,
                                             "illegal": run.illegal_policy_actions, "seconds": round(dt, 1)}
            log(f"{student:26s} {arm:8s} {len(out)} games  EWR {ewr:.4f}  errors {run.policy_errors}  {dt:.0f}s")

    # ---- analysis: paired, family-stratified bootstrap (the G3 machinery, unchanged) --------
    def ns(rowdicts, offset=0):
        return [SimpleNamespace(match_id=o["match_id"], setup_pair_id=o["setup_pair_id"] + offset,
                                candidate_score=o["candidate_score"], errored=o["errored"]) for o in rowdicts]

    contrasts = [("learned", "frozen"), ("learned", "library"), ("frozen", "library")]
    analysis = {"per_student": {}, "pooled": {}, "ewr": {}}
    for student in HANDCRAFTED_OPPONENTS:
        analysis["ewr"][student] = {arm: sum(o["candidate_score"] for o in rows[(student, arm)]) / 2560 for arm in ("learned", "frozen", "library")}
        analysis["per_student"][student] = {}
        for a, b in contrasts:
            res = ev.paired_analysis(ns(rows[(student, a)]), ns(rows[(student, b)]), cases, namespace=NAMESPACE)
            analysis["per_student"][student][f"{a}-{b}"] = {k: res[k] for k in ("point", "lower", "upper", "candidate_ewr", "control_ewr", "cases_on_which_arms_differ", "bootstrap_standard_error")}
    # pooled across the 8 students: offset the case indices so every base carries 8 x 16 cases
    pooled_cases = []
    for s, student in enumerate(HANDCRAFTED_OPPONENTS):
        for c in cases:
            pooled_cases.append(dataclasses.replace(c, case_index=s * 2560 + c.case_index))
    for a, b in contrasts:
        ra, rb = [], []
        for s, student in enumerate(HANDCRAFTED_OPPONENTS):
            ra += ns(rows[(student, a)], s * 2560); rb += ns(rows[(student, b)], s * 2560)
        res = ev.paired_analysis(ra, rb, pooled_cases, namespace=NAMESPACE)
        analysis["pooled"][f"{a}-{b}"] = {k: res[k] for k in ("point", "lower", "upper", "candidate_ewr", "control_ewr", "cases", "bases", "families", "cases_on_which_arms_differ", "bootstrap_standard_error", "by_opponent", "by_colour")}

    analysis["setup_model_digests"] = digests
    analysis["own_setups_differ_learned_vs_frozen"] = int(sum(a != b for a, b in zip(fp_learned, fp_frozen)))
    analysis["design"] = __doc__
    analysis["game_summaries"] = summaries
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1, sort_keys=True) + "\n")
    log("DONE")


if __name__ == "__main__":
    main()
