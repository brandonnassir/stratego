"""Baselines on the frozen G3 schedule (2,560 paired games vs 8 handcrafted opponents, battleless 200, greedy, G3 seeds):
  untrained      : G3 control bundle_0000 — untrained C1 + untrained setup model 082ff778 (evaluate_bundle, same-bundle)
  phase8@library : the accepted Phase 8 warm-start C1 (24,000 updates, corpus only) + same-family library formations
  phase8@ckpt1024: Phase 8 C1 + the from-scratch setup model's period-1,024 formations
  phase8@g3init  : Phase 8 C1 + G3's init setup model formations (the formations control_final played)
"""
import importlib.util, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch; torch.set_num_threads(2)
ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent"); SRC = ROOT / "output/phase18/worktrees/g3-stage6b"; sys.path.insert(0, str(SRC))
RT = ROOT / "output/phase18/runtime/g3_pilot_v2"; FS = ROOT / "output/phase18/runtime/addendum_library_setup_from_scratch"; W = ROOT / "output/phase18/runtime/addendum_baselines"
from stratego.training.phase18 import g3_evaluation as ev
from stratego.training.phase18.setup_contract import SetupTrainingConfig
from stratego.training.phase18.setup_learning import SetupTrainer
from stratego.training.phase18.setup_model import state_dict_digest
from stratego.training.phase18.g3_bundle import load_setup_trainer
from stratego.evaluation.neural_worker import InferenceOwner, run_neural_schedule
from stratego.model.architecture_configs import ARCHITECTURE_FAMILY, candidate_config
spec = importlib.util.spec_from_file_location("driver", SRC / "scripts/phase18_g3_pilot.py"); driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
NAMESPACE = driver.NAMESPACE; NS = "phase18_addendum_library_setup_from_scratch_v1"
def log(m):
    line = f"[baselines {time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(W / "run.log", "a") as f: f.write(line + "\n")
def main():
    cases = ev.build_cases(ev.load_evaluation_bases()); bases = ev.load_evaluation_bases(); by_key = {(e.family_id, e.base_index): e for e in bases}
    g3cfg = driver.production_config("control", c1_device="mps")
    rows = {}
    # 1. untrained baseline through the production evaluator (bundle_0000 couples the untrained C1 with the untrained setup)
    if not (W / "rows_untrained.jsonl").exists():
        rec, res = ev.evaluate_bundle(RT / "control/bundles/bundle_0000", config=g3cfg, lineage="control", label="untrained", cases=cases, work=W / "untrained", device="mps", workers=8, log=log)
        out = [{"match_id": r.match_id, "setup_pair_id": r.setup_pair_id, "candidate_score": r.candidate_score, "errored": bool(r.errored)} for r in res]
        (W / "rows_untrained.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out)); log(f"untrained: C1 {rec['c1_state_digest'][:12]} setup {rec['setup_model_digest'][:12]} EWR {sum(o['candidate_score'] for o in out)/len(out):.4f}")
    rows["untrained"] = [json.loads(l) for l in (W / "rows_untrained.jsonl").read_text().splitlines() if l.strip()]
    # 2-4. the Phase 8 warm-start C1 with three formation sources
    ctrl_export = ev.export_bundle_c1(RT / "control/bundles/bundle_0256", W / "control_c1.pt")
    p8 = ROOT / "checkpoints/phase8/agent07/warmstart_eval.pt"
    owner = InferenceOwner(p8, decision_mode=ev.DECISION_MODE_GREEDY, device="mps", dtype=ev.GATE_DTYPE, expected_architecture_id=ARCHITECTURE_FAMILY, expected_configuration=candidate_config(ctrl_export["candidate_id"]), name="phase8_warmstart")
    log(f"Phase 8 warm-start export loaded: {owner.identity()}")
    banks = {}
    tr, _ = SetupTrainer.load_checkpoint(FS / "ckpt_1024", SetupTrainingConfig(run_id="ADDENDUM-LIB-SETUP-2026-A", device="cpu", pool_size=1024), namespace=NS, seed_index=1, device="cpu")
    banks["ckpt1024"] = ev.build_arm_bank(cases, ev.resolve_own_setups(tr.evaluation_model(device="cpu"), cases, namespace=NAMESPACE, seed_index=1, device="cpu").samples)
    tr0, _ = load_setup_trainer(RT / "control/bundles/bundle_0256", g3cfg, device="cpu"); m0 = tr0.evaluation_model(device="cpu"); assert state_dict_digest(m0).startswith("082ff778")
    banks["g3init"] = ev.build_arm_bank(cases, ev.resolve_own_setups(m0, cases, namespace=NAMESPACE, seed_index=1, device="cpu").samples)
    pairs = []
    for case in cases:
        own = by_key[(case.family_id, 400 + ((case.base_index - 400 + 5) % 10))]
        own_e = ev.validate_setup(ev.orient_setup(tuple(int(v) for v in own.canonical_setup), case.colour), case.colour)
        oc = ev.RED if case.colour == ev.BLUE else ev.BLUE; base_e = ev.validate_setup(ev.orient_setup(case.base_canonical, oc), oc)
        red, blue = (own_e, base_e) if case.colour == ev.RED else (base_e, own_e)
        pairs.append(ev.SetupPair(setup_pair_id=int(case.case_index), red_setup=red, blue_setup=blue, generation_seed=0, bank_version=ev.G3_EVALUATION_BANK_VERSION, generation_family="addendum_library_same_family_rot5"))
    banks["library"] = ev.SetupBank(bank_version=ev.G3_EVALUATION_BANK_VERSION, root_seed=0, generation_family="addendum_library_same_family_rot5", pairs=tuple(pairs))
    cref = ev.candidate_policy_ref()
    try:
        for arm in ("library", "ckpt1024", "g3init"):
            path = W / f"rows_phase8_{arm}.jsonl"
            if path.exists(): rows[f"phase8@{arm}"] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]; continue
            matches = ev.build_schedule(cases, namespace=NAMESPACE)
            for mm in matches: mm.resolve_setups(banks[arm])
            def runner(chunk, bank=banks[arm]):
                run = run_neural_schedule(chunk, bank, owner, policy_ref=cref, worker_count=8, record_actions=False, on_policy_error=ev.ON_POLICY_ERROR_QUARANTINE)
                return run.results, {"policy_errors": run.policy_errors}
            t = time.perf_counter(); results, reports = ev.play_chunks(matches, W / f"phase8_{arm}" / "games", runner, chunk_units=64, label=f"phase8@{arm}", log=log)
            acc = ev.reconcile(matches, results)
            out = [{"match_id": r.match_id, "setup_pair_id": r.setup_pair_id, "candidate_score": r.candidate_score, "errored": bool(r.errored)} for r in results]
            path.write_text("".join(json.dumps(o) + "\n" for o in out)); rows[f"phase8@{arm}"] = out
            log(f"phase8@{arm}: EWR {sum(o['candidate_score'] for o in out)/len(out):.4f} errors {sum(r['policy_errors'] for r in reports)} reconciles {acc['reconciles']} {time.perf_counter()-t:.0f}s")
    finally:
        owner.close()
    (W / "summary.json").write_text(json.dumps({k: sum(o["candidate_score"] for o in v) / len(v) for k, v in rows.items()}, indent=1) + "\n"); log("BASELINES DONE")
if __name__ == "__main__":
    main()
