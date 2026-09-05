#!/usr/bin/env python3
"""Operator addendum v3: do the from-scratch model's formations help the NEURAL mover?

One fixed neural mover — the G3 control lineage's final C1 (bundle e1c410021f71, a
policy that never co-adapted to any learned formation) — plays the frozen G3 schedule
(2,560 paired games) with its own formation from four sources:

  g3_init    the G3 init setup model 082ff778: this IS control_final; its receipts are
             reused (same C1, same schedule, same seeds, byte-identical formations)
  ckpt_0     the from-scratch model's init c549bc02 (untrained)
  ckpt_1024  the from-scratch model after 1,024 periods of handcrafted-game learning
  library    a same-family handcrafted library formation (rotated; never the opponent's)

Everything else is the G3 evaluation: greedy decisions, float32, battleless 200, the G3
case seeds for own-setup sampling, the G3 paired family-stratified bootstrap.
"""
import importlib.util, json, sys, time, argparse
from pathlib import Path
from types import SimpleNamespace
import torch
torch.set_num_threads(2)
ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
SRC = ROOT / "output/phase18/worktrees/g3-stage6b"; sys.path.insert(0, str(SRC))
RT = ROOT / "output/phase18/runtime/g3_pilot_v2"
FS = ROOT / "output/phase18/runtime/addendum_library_setup_from_scratch"
W = ROOT / "output/phase18/runtime/addendum_neural_mover"
from stratego.training.phase18 import g3_evaluation as ev
from stratego.training.phase18.setup_contract import SetupTrainingConfig
from stratego.training.phase18.setup_learning import SetupTrainer
from stratego.training.phase18.setup_model import state_dict_digest
from stratego.evaluation.neural_worker import InferenceOwner, run_neural_schedule
from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
spec = importlib.util.spec_from_file_location("driver", SRC / "scripts/phase18_g3_pilot.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
NAMESPACE = driver.NAMESPACE
NS = "phase18_addendum_library_setup_from_scratch_v1"; RUN_ID = "ADDENDUM-LIB-SETUP-2026-A"

def log(m):
    line = f"[neural {time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(W / "run.log", "a") as f: f.write(line + "\n")

def library_bank(cases, by_key):
    pairs = []
    for case in cases:
        own = by_key[(case.family_id, 400 + ((case.base_index - 400 + 5) % 10))]
        own_e = ev.validate_setup(ev.orient_setup(tuple(int(v) for v in own.canonical_setup), case.colour), case.colour)
        oc = ev.RED if case.colour == ev.BLUE else ev.BLUE
        base_e = ev.validate_setup(ev.orient_setup(case.base_canonical, oc), oc)
        red, blue = (own_e, base_e) if case.colour == ev.RED else (base_e, own_e)
        pairs.append(ev.SetupPair(setup_pair_id=int(case.case_index), red_setup=red, blue_setup=blue, generation_seed=0,
                                  bank_version=ev.G3_EVALUATION_BANK_VERSION, generation_family="addendum_library_same_family_rot5"))
    return ev.SetupBank(bank_version=ev.G3_EVALUATION_BANK_VERSION, root_seed=0, generation_family="addendum_library_same_family_rot5", pairs=tuple(pairs))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true"); a = p.parse_args()
    W.mkdir(parents=True, exist_ok=True)
    bases = ev.load_evaluation_bases(); cases = ev.build_cases(bases); by_key = {(e.family_id, e.base_index): e for e in bases}
    if a.smoke: cases = cases[:32]
    # the fixed neural mover: the G3 control lineage's final C1
    bundle = RT / "control/bundles/bundle_0256"
    export = ev.export_bundle_c1(bundle, W / "eval_weights.pt")
    control_arm = json.loads((RT / "evaluation/control_final/arm_record.json").read_text())
    assert export["c1_state_digest"] == control_arm["c1_state_digest"], "the exported C1 is not the one that played control_final"
    log(f"neural mover: control bundle_0256 C1 {export['c1_state_digest'][:12]} (== control_final's mover)")
    config = SetupTrainingConfig(run_id=RUN_ID, device="cpu", pool_size=1024)
    banks = {}
    for arm, ck in (("ckpt_0", 0), ("ckpt_1024", 1024)):
        tr, _ = SetupTrainer.load_checkpoint(FS / f"ckpt_{ck:04d}", config, namespace=NS, seed_index=1, device="cpu")
        model = tr.evaluation_model(device="cpu")
        gen = ev.resolve_own_setups(model, cases, namespace=NAMESPACE, seed_index=1, device="cpu")
        banks[arm] = ev.build_arm_bank(cases, gen.samples); log(f"{arm}: setup EMA {state_dict_digest(model)[:12]}, {gen.telemetry['distinct_content_fingerprints']} distinct formations")
    banks["library"] = library_bank(cases, by_key)
    owner = InferenceOwner(W / "eval_weights.pt", decision_mode=ev.DECISION_MODE_GREEDY, device="mps", dtype=ev.GATE_DTYPE,
                           expected_architecture_id=ARCHITECTURE_FAMILY, expected_configuration=candidate_config(export["candidate_id"]), name="addendum_neural_mover")
    cref = ev.candidate_policy_ref()
    rows = {}
    try:
        for arm in (["ckpt_1024"] if a.smoke else ["ckpt_1024", "library", "ckpt_0"]):
            path = W / f"rows_{arm}.jsonl"
            if path.exists() and not a.smoke:
                rows[arm] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]; log(f"{arm}: reused {len(rows[arm])} rows"); continue
            matches = ev.build_schedule(cases, namespace=NAMESPACE)
            for m in matches: m.resolve_setups(banks[arm])
            def runner(chunk, bank=banks[arm]):
                run = run_neural_schedule(chunk, bank, owner, policy_ref=cref, worker_count=8, record_actions=False, on_policy_error=ev.ON_POLICY_ERROR_QUARANTINE)
                return run.results, {"matches": run.matches_run, "policy_errors": run.policy_errors, "illegal": run.illegal_policy_actions, "seconds": round(run.wall_clock_seconds, 1)}
            t = time.perf_counter()
            results, reports = ev.play_chunks(matches, W / arm / "games", runner, chunk_units=64, label=arm, log=log)
            acc = ev.reconcile(matches, results)
            out = [{"match_id": r.match_id, "setup_pair_id": r.setup_pair_id, "candidate_score": r.candidate_score, "errored": bool(r.errored), "plies": r.plies, "terminal_reason": r.terminal_reason} for r in results]
            if not a.smoke: path.write_text("".join(json.dumps(o) + "\n" for o in out))
            rows[arm] = out
            log(f"{arm}: {len(out)} games, EWR {sum(o['candidate_score'] for o in out)/len(out):.4f}, errors {sum(r['policy_errors'] for r in reports)}, reconciles {acc['reconciles']}, {time.perf_counter()-t:.0f}s")
    finally:
        owner.close()
    if a.smoke: log("SMOKE OK"); return
    # the g3_init arm is control_final itself
    g3 = ev.read_receipt_rows(RT / "evaluation/control_final/receipts.jsonl")
    rows["g3_init"] = [{"match_id": r.match_id, "setup_pair_id": r.setup_pair_id, "candidate_score": r.candidate_score, "errored": bool(r.errored)} for r in g3]
    def ns(rd): return [SimpleNamespace(match_id=o["match_id"], setup_pair_id=o["setup_pair_id"], candidate_score=o["candidate_score"], errored=o["errored"]) for o in rd]
    analysis = {"mover": {"bundle": "control/bundle_0256", "c1_state_digest": export["c1_state_digest"]},
                "ewr": {arm: sum(o["candidate_score"] for o in rows[arm]) / len(rows[arm]) for arm in rows}, "contrasts": {}}
    for a_, b_ in (("ckpt_1024", "g3_init"), ("ckpt_1024", "ckpt_0"), ("ckpt_1024", "library"), ("library", "g3_init"), ("ckpt_0", "g3_init")):
        r = ev.paired_analysis(ns(rows[a_]), ns(rows[b_]), cases, namespace=NAMESPACE)
        analysis["contrasts"][f"{a_}-{b_}"] = {k: r[k] for k in ("point", "lower", "upper", "candidate_ewr", "control_ewr", "cases_on_which_arms_differ", "bootstrap_standard_error", "by_opponent", "by_colour")}
    (W / "analysis.json").write_text(json.dumps(analysis, indent=1, sort_keys=True) + "\n"); log("NEURAL EVAL DONE")

if __name__ == "__main__":
    main()
