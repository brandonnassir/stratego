#!/usr/bin/env python3
"""Operator addendum v4: let the neural policy CO-ADAPT to the good formations.

A G3 control-style lineage (setup model FROZEN, C1 trained on the canonical/live mixture)
whose frozen setup model is the from-scratch model's period-1,024 EMA (c0f532e7...)
instead of G3's random init. Everything else is G3's control lineage: the same C1 init
(canonical Phase 8 seed), the same collector and C1 seeds (G3's namespace), the same
256-period budget, the production LineageRunner untouched. Only the formations the
policy trains on differ. Its final C1 is then evaluated with ckpt_1024 formations against
the un-adapted G3 control C1 with the same formations (addendum v3: 0.7324).
"""
import importlib.util, json, sys, time, argparse, dataclasses
from pathlib import Path
ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
SRC = ROOT / "output/phase18/worktrees/g3-stage6b"; sys.path.insert(0, str(SRC)); sys.path.insert(0, str(SRC / "scripts"))
FS = ROOT / "output/phase18/runtime/addendum_library_setup_from_scratch"
W = ROOT / "output/phase18/runtime/addendum_coadapt"
RUNTIME = W / "runtime"
spec = importlib.util.spec_from_file_location("driver", SRC / "scripts/phase18_g3_pilot.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
from stratego.training.phase18 import g3_pilot as gp
from stratego.training.phase18.setup_contract import SetupTrainingConfig
from stratego.training.phase18.setup_learning import SetupTrainer
from stratego.training.phase18.setup_model import state_dict_digest, build_setup_model as _real_build
from stratego.training.warmstart_checkpoint import verify_corpus_identity
RUN_ID = "ADDENDUM-COADAPT-2026-A"
FS_NS = "phase18_addendum_library_setup_from_scratch_v1"

def log(m):
    line = f"[coadapt {time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(W / "logs/lineage.log", "a") as f: f.write(line + "\n")

def frozen_setup_state():
    tr, _ = SetupTrainer.load_checkpoint(FS / "ckpt_1024", SetupTrainingConfig(run_id="ADDENDUM-LIB-SETUP-2026-A", device="cpu", pool_size=1024), namespace=FS_NS, seed_index=1, device="cpu")
    ema = tr.evaluation_model(device="cpu"); d = state_dict_digest(ema)
    assert d.startswith("c0f532e7"), d
    return {k: v.detach().clone() for k, v in ema.state_dict().items()}, d

def main():
    p = argparse.ArgumentParser(); p.add_argument("--resume", action="store_true"); p.add_argument("--periods", type=int, default=None); a = p.parse_args()
    state, digest = frozen_setup_state()
    # The one redirection: every setup model this lineage constructs starts as ckpt_1024's EMA.
    def build_from_ckpt(device="cpu", seed=None):
        m = _real_build(device=device); m.load_state_dict({k: v.to(device) for k, v in state.items()}); return m
    gp.build_setup_model = build_from_ckpt
    config = dataclasses.replace(driver.production_config("control", c1_device="mps"), run_id=RUN_ID)
    assert not config.setup_updates_enabled
    root, accepted = driver.accepted_corpus()
    log("verifying the accepted corpus identity and every payload")
    identity = verify_corpus_identity(root, accepted, check_payload_bytes=True)
    kw = dict(run_root=RUNTIME, corpus_root=root, corpus_identity=identity, log=log)
    t0 = time.perf_counter()
    if a.resume:
        runner = gp.LineageRunner.resume(config, bundle_directory=None, **kw)
        sel = runner.resume_record["selection"]; log(f"resumed from {Path(sel['bundle']).name} at period {runner.period}")
        if runner.resume_record["archive"]["archive"]: log(f"archived later progress under {runner.resume_record['archive']['archive']}")
    else:
        runner = gp.LineageRunner.fresh(config, **kw)
        log(f"fresh lineage: setup init digest {runner.init_record['setup_init_state_digest'][:12]} (ckpt_1024 EMA {digest[:12]}), C1 init {runner.init_record['c1_init_state_digest'][:12]}")
        assert runner.init_record["setup_init_state_digest"] == digest
    try:
        records = runner.run(periods=a.periods)
    finally:
        runner.close()
    summary = {"run_id": RUN_ID, "lineage": "control", "frozen_setup_model": "from-scratch ckpt_1024 EMA " + digest, "namespace": config.namespace,
               "periods_run_this_process": len(records), "period_reached": runner.period, "complete": runner.period == config.periods,
               "last_bundle_id": runner.last_bundle_id, "last_bundle_period": runner.last_bundle_period, "integrity": runner.integrity,
               "setup_skips": runner.setup_skips, "wall_seconds": round(time.perf_counter() - t0, 1)}
    (RUNTIME / "control/run_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    log(f"period {runner.period}/{config.periods} in {summary['wall_seconds']:.0f} s" + ("; LINEAGE DONE" if summary["complete"] else ""))

if __name__ == "__main__":
    main()
