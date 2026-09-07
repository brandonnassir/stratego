#!/usr/bin/env python3
"""Addendum v4 evaluation at period 224 (budget-matched).
Movers: the co-adapted C1 (this lineage, bundle_0224, trained on ckpt_1024 formations) and
G3's control C1 at ITS bundle_0224 (trained on the random-init formations). Same C1 init,
same seeds, same 14,336 updates; only the training-game formations differ. Frozen 2,560-game
G3 schedule, G3 seeds; own formation from ckpt_1024 / library / g3_init."""
import importlib.util, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch; torch.set_num_threads(2)
ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
SRC = ROOT / "output/phase18/worktrees/g3-stage6b"; sys.path.insert(0, str(SRC))
RT = ROOT / "output/phase18/runtime/g3_pilot_v2"; FS = ROOT / "output/phase18/runtime/addendum_library_setup_from_scratch"
V3 = ROOT / "output/phase18/runtime/addendum_neural_mover"; W = ROOT / "output/phase18/runtime/addendum_coadapt"
from stratego.training.phase18 import g3_evaluation as ev
from stratego.training.phase18.setup_contract import SetupTrainingConfig
from stratego.training.phase18.setup_learning import SetupTrainer
from stratego.training.phase18.setup_model import state_dict_digest
from stratego.training.phase18.g3_bundle import read_manifest, load_setup_trainer
from stratego.evaluation.neural_worker import InferenceOwner, run_neural_schedule
from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
spec = importlib.util.spec_from_file_location("driver", SRC / "scripts/phase18_g3_pilot.py"); driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
NAMESPACE = driver.NAMESPACE; NS = "phase18_addendum_library_setup_from_scratch_v1"; P = int(sys.argv[1]) if len(sys.argv) > 1 else 375

def log(m):
    line = f"[eval-p375 {time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(W / "logs/eval_p375.log", "a") as f: f.write(line + "\n")

def ns(rd): return [SimpleNamespace(match_id=o["match_id"], setup_pair_id=o["setup_pair_id"], candidate_score=o["candidate_score"], errored=o["errored"]) for o in rd]

def main():
    bases = ev.load_evaluation_bases(); cases = ev.build_cases(bases); by_key = {(e.family_id, e.base_index): e for e in bases}
    banks = {}
    tr, _ = SetupTrainer.load_checkpoint(FS / "ckpt_1024", SetupTrainingConfig(run_id="ADDENDUM-LIB-SETUP-2026-A", device="cpu", pool_size=1024), namespace=NS, seed_index=1, device="cpu")
    gen = ev.resolve_own_setups(tr.evaluation_model(device="cpu"), cases, namespace=NAMESPACE, seed_index=1, device="cpu"); banks["ckpt_1024"] = ev.build_arm_bank(cases, gen.samples)
    g3cfg = driver.production_config("control", c1_device="mps")
    tr0, _ = load_setup_trainer(RT / "control/bundles/bundle_0256", g3cfg, device="cpu"); m0 = tr0.evaluation_model(device="cpu"); assert state_dict_digest(m0).startswith("082ff778")
    gen = ev.resolve_own_setups(m0, cases, namespace=NAMESPACE, seed_index=1, device="cpu"); banks["g3_init"] = ev.build_arm_bank(cases, gen.samples)
    pairs = []
    for case in cases:
        own = by_key[(case.family_id, 400 + ((case.base_index - 400 + 5) % 10))]
        own_e = ev.validate_setup(ev.orient_setup(tuple(int(v) for v in own.canonical_setup), case.colour), case.colour)
        oc = ev.RED if case.colour == ev.BLUE else ev.BLUE; base_e = ev.validate_setup(ev.orient_setup(case.base_canonical, oc), oc)
        red, blue = (own_e, base_e) if case.colour == ev.RED else (base_e, own_e)
        pairs.append(ev.SetupPair(setup_pair_id=int(case.case_index), red_setup=red, blue_setup=blue, generation_seed=0, bank_version=ev.G3_EVALUATION_BANK_VERSION, generation_family="addendum_library_same_family_rot5"))
    banks["library"] = ev.SetupBank(bank_version=ev.G3_EVALUATION_BANK_VERSION, root_seed=0, generation_family="addendum_library_same_family_rot5", pairs=tuple(pairs))
    movers = {"coadapt_p375": W / ("runtime/control/bundles/bundle_%04d" % P)}
    plan = [("coadapt_p375", "ckpt_1024"), ("coadapt_p375", "library"), ("coadapt_p375", "g3_init")]
    rows, digests = {}, {}
    cref = ev.candidate_policy_ref()
    for mover, arm in plan:
        path = W / f"rows_{mover}_{arm}.jsonl"
        if path.exists(): rows[(mover, arm)] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]; continue
        man = read_manifest(movers[mover]); export = ev.export_bundle_c1(movers[mover], W / f"eval_weights_{mover}.pt"); digests[mover] = export["c1_state_digest"]
        assert int(man["period"]) == P and man["components"]["c1"]["global_step"] == 64 * P, (man["period"], man["components"]["c1"]["global_step"])
        owner = InferenceOwner(W / f"eval_weights_{mover}.pt", decision_mode=ev.DECISION_MODE_GREEDY, device="mps", dtype=ev.GATE_DTYPE, expected_architecture_id=ARCHITECTURE_FAMILY, expected_configuration=candidate_config(export["candidate_id"]), name=f"eval_{mover}")
        try:
            matches = ev.build_schedule(cases, namespace=NAMESPACE)
            for mm in matches: mm.resolve_setups(banks[arm])
            def runner(chunk, bank=banks[arm], owner=owner):
                run = run_neural_schedule(chunk, bank, owner, policy_ref=cref, worker_count=8, record_actions=False, on_policy_error=ev.ON_POLICY_ERROR_QUARANTINE)
                return run.results, {"policy_errors": run.policy_errors}
            t = time.perf_counter(); results, reports = ev.play_chunks(matches, W / "eval_p375" / mover / arm / "games", runner, chunk_units=64, label=f"{mover}@{arm}", log=log)
            acc = ev.reconcile(matches, results)
        finally:
            owner.close()
        out = [{"match_id": r.match_id, "setup_pair_id": r.setup_pair_id, "candidate_score": r.candidate_score, "errored": bool(r.errored)} for r in results]
        path.write_text("".join(json.dumps(o) + "\n" for o in out)); rows[(mover, arm)] = out
        log(f"{mover}@{arm}: C1 {digests[mover][:12]} step {man['components']['c1']['global_step']}, EWR {sum(o['candidate_score'] for o in out)/len(out):.4f}, errors {sum(r['policy_errors'] for r in reports)}, reconciles {acc['reconciles']}, {time.perf_counter()-t:.0f}s")
        analyse(rows, cases, digests)
    log("EVAL P375 DONE")

def analyse(rows, cases, digests):
    B = ROOT / "output/phase18/runtime/addendum_baselines"
    def load(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    p8 = {"ckpt_1024": load(B / "rows_phase8_ckpt1024.jsonl"), "library": load(B / "rows_phase8_library.jsonl"), "g3_init": load(B / "rows_phase8_g3init.jsonl")}
    v3 = {"ckpt_1024": load(V3 / "rows_ckpt_1024.jsonl"), "library": load(V3 / "rows_library.jsonl")}
    v3["g3_init"] = [{"match_id": r.match_id, "setup_pair_id": r.setup_pair_id, "candidate_score": r.candidate_score, "errored": bool(r.errored)} for r in ev.read_receipt_rows(RT / "evaluation/control_final/receipts.jsonl")]
    v4 = {a: load(W / f"rows_coadapt_p224_{a}.jsonl") for a in ("ckpt_1024", "library", "g3_init")}
    A = {"movers": digests, "ewr": {f"{m}@{a}": sum(o["candidate_score"] for o in v)/len(v) for (m, a), v in rows.items()},
         "ewr_phase8_24k": {a: sum(o["candidate_score"] for o in v)/len(v) for a, v in p8.items()}, "ewr_g3control_p256": {a: sum(o["candidate_score"] for o in v)/len(v) for a, v in v3.items()},
         "ewr_coadapt_p224": {a: sum(o["candidate_score"] for o in v)/len(v) for a, v in v4.items()}, "contrasts": {}}
    def C(label, ra, rb):
        r = ev.paired_analysis(ns(ra), ns(rb), cases, namespace=NAMESPACE); A["contrasts"][label] = {k: r[k] for k in ("point", "lower", "upper", "candidate_ewr", "control_ewr", "cases_on_which_arms_differ", "by_opponent")}
    for arm in ("ckpt_1024", "library", "g3_init"):
        if ("coadapt_p375", arm) in rows:
            C(f"MATCHED BUDGET: coadapt_24k - phase8_24k @ {arm}", rows[("coadapt_p375", arm)], p8[arm])
            C(f"coadapt_24k - g3control_p256 @ {arm}", rows[("coadapt_p375", arm)], v3[arm])
            C(f"coadapt_24k - coadapt_p224 @ {arm} (224->375 gain)", rows[("coadapt_p375", arm)], v4[arm])
    if ("coadapt_p375", "ckpt_1024") in rows and ("coadapt_p375", "library") in rows: C("coadapt_24k: ckpt_1024 - library", rows[("coadapt_p375", "ckpt_1024")], rows[("coadapt_p375", "library")])
    (W / "analysis_p375.json").write_text(json.dumps(A, indent=1, sort_keys=True) + "\n")

if __name__ == "__main__":
    main()
