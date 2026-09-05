#!/usr/bin/env python3
"""Operator addendum v2: train a FRESH setup model from scratch on handcrafted games only.

Every period: draw formations uniformly WITH REPLACEMENT from the 6,400 training-split
library formations (indices 0..399 of all 16 families); each of the 8 handcrafted bots
plays every other bot (56 ordered pairs, cycled; no self-play); the setup model observes
those games and learns from the outcomes attributed to the formations used. The model
never plays and never generates its own training formations.

The learning rule is the G3 setup trainer, byte-for-byte: the same model, PPO-clip policy
loss, value head, entropy-prediction head, behaviour KL, alpha schedule, EMA and
checkpoint format. Library formations enter the buffer as teacher-forced SampledSetup
rows scored under the current raw model, so the behaviour log-probabilities are the
model's own at draw time.
"""
import json, sys, time, argparse
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)

ROOT = Path("/Users/brandonwashington/Dev/Github/stratego/gpt_agent")
SRC = ROOT / "output/phase18/worktrees/g3-stage6b"
sys.path.insert(0, str(SRC))
OUT = ROOT / "output/phase18/runtime/addendum_library_setup_from_scratch"

from stratego.engine.constants import FLAG, RED, BLUE
from stratego.setups.identity import CANONICAL_FILES, content_fingerprint
from stratego.setups.library import read_library_jsonl
from stratego.training.phase18 import g3_evaluation as ev
from stratego.training.phase18.g3_contract import HANDCRAFTED_OPPONENTS, COLLECTOR_RULES, setup_model_init_seed
from stratego.training.phase18.setup_contract import (FLAG_PERMITTED_FILES, SETUP_PREFIXES, SETUP_SEQUENCE_LENGTH,
                                                       START_TOKEN, SetupTrainingConfig, stream_seed)
from stratego.training.phase18.setup_model import build_setup_model, state_dict_digest
from stratego.training.phase18.setup_learning import SetupTrainer
from stratego.training.phase18.setup_buffer import SetupBuffer
from stratego.training.phase18.setup_sampling import (SampledSetup, legal_masks, masked_log_probabilities,
                                                       reflect_tokens, suffix_information, to_engine_setup)
from stratego.evaluation.registry import policy_ref
from stratego.evaluation.match_spec import MatchSpec, SetupBank, SetupPair
from stratego.evaluation.match_runner import run_schedule

NS = "phase18_addendum_library_setup_from_scratch_v1"
SEED_INDEX = 1
RUN_ID = "ADDENDUM-LIB-SETUP-2026-A"
PERIODS = 256
GAMES_PER_PERIOD = 2048
CHECKPOINT_EVERY = 32
WORKERS = 8
BANK_VERSION = "addendum_library_training_bank_v1"

def log(msg):
    line = f"[train {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "train.log", "a") as f: f.write(line + "\n")

# ---- teacher-forced rows for library formations -----------------------------------------
@torch.no_grad()
def score_formations(model, canonicals, *, snapshot_digest, period):
    model.eval()
    B = len(canonicals)
    played = np.array(canonicals, dtype=np.int64)                                   # [B, 40]
    flag_file = np.array([int(np.nonzero(r == FLAG)[0][0]) % CANONICAL_FILES for r in played])
    reflected = np.array([f not in FLAG_PERMITTED_FILES for f in flag_file])
    network = played.copy()
    if reflected.any(): network[reflected] = reflect_tokens(played[reflected])
    tokens = torch.as_tensor(network, dtype=torch.long)
    sequence = torch.full((B, SETUP_SEQUENCE_LENGTH), START_TOKEN, dtype=torch.long); sequence[:, 1:] = tokens
    out = model(sequence)
    masks = np.zeros((B, SETUP_PREFIXES, 12), dtype=bool); logp = np.zeros((B, SETUP_PREFIXES, 12), dtype=np.float32)
    selected = np.zeros((B, SETUP_PREFIXES), dtype=np.float32); ar = np.arange(B)
    for prefix in range(SETUP_PREFIXES):
        mask = legal_masks(tokens, prefix, force_handedness=True)
        step = masked_log_probabilities(out["piece_logits"][:, prefix], mask)
        chosen = tokens[:, prefix]
        m = mask.numpy(); s = step.numpy()
        if not m[ar, chosen.numpy()].all(): raise RuntimeError(f"a library formation is illegal under the mask at prefix {prefix}")
        masks[:, prefix] = m; logp[:, prefix] = np.where(m, s, 0.0); selected[:, prefix] = s[ar, chosen.numpy()]
    wdl = out["wdl_logits"].to(torch.float32).numpy(); ent = out["entropy_prediction"].to(torch.float32).numpy()
    rows = []
    for i in range(B):
        pc = tuple(int(v) for v in played[i])
        rows.append(SampledSetup(index=i, lane=RED, root_seed=0, reflection_seed=0, reflected=bool(reflected[i]),
                                 network_tokens=network[i].astype(np.int8), played_canonical=pc,
                                 engine_setup=to_engine_setup(pc, RED), legal_masks=masks[i], behavior_log_probs=logp[i],
                                 behavior_selected_log_prob=selected[i], suffix_information=suffix_information(selected[i]),
                                 wdl_logits=wdl[i], entropy_prediction=ent[i], snapshot_digest=snapshot_digest,
                                 snapshot_iteration=int(period)))
    return rows, int(reflected.sum())

def main():
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true"); p.add_argument("--periods", type=int, default=PERIODS); a = p.parse_args()
    periods, games = (1, 64) if a.smoke else (a.periods, GAMES_PER_PERIOD)
    OUT.mkdir(parents=True, exist_ok=True)
    library = [e for e in read_library_jsonl(ev.LIBRARY_JSONL_PATH) if e.base_index < 400]
    canon = [tuple(int(v) for v in e.canonical_setup) for e in library]
    fps = [content_fingerprint(c) for c in canon]
    assert len(canon) == 6400 and len(set(fps)) == 6400
    bots = list(HANDCRAFTED_OPPONENTS)
    cells = [(r, b) for r in bots for b in bots if r != b]
    assert len(cells) == 56
    config = SetupTrainingConfig(run_id=RUN_ID, device="cpu", pool_size=1024)

    state_path = OUT / "state.json"
    if state_path.exists() and not a.smoke:
        st = json.loads(state_path.read_text()); start = st["period"] + 1
        trainer, _ = SetupTrainer.load_checkpoint(OUT / f"ckpt_{st['period']:04d}", config, namespace=NS, seed_index=SEED_INDEX, device="cpu")
        log(f"resumed from ckpt_{st['period']:04d} (raw {state_dict_digest(trainer.model)[:12]})")
    else:
        model = build_setup_model(device="cpu", seed=setup_model_init_seed(NS, SEED_INDEX))
        trainer = SetupTrainer(model, config, namespace=NS, seed_index=SEED_INDEX); start = 1
        if not a.smoke:
            trainer.save_checkpoint(OUT / "ckpt_0000")
        log(f"fresh init raw {state_dict_digest(model)[:12]} ema {state_dict_digest(trainer.ema.as_model(device='cpu'))[:12]} | {len(canon)} library formations | {len(cells)} cells")
    buffer = SetupBuffer(storage_duration=21, device="cpu")

    for period in range(start, periods + 1):
        t0 = time.perf_counter()
        rng = np.random.default_rng(stream_seed(NS, "library_draw", SEED_INDEX, period) % (2**63))
        draws = rng.integers(0, len(canon), size=2 * games)                       # with replacement
        distinct = sorted(set(int(d) for d in draws))
        snap = state_dict_digest(trainer.model)
        rows, n_refl = score_formations(trainer.generation_actor, [canon[i] for i in distinct], snapshot_digest=snap, period=period)
        pool = buffer.add_pool(rows, period=period)
        t_score = time.perf_counter() - t0
        # play: game k uses cell k % 56, red formation draws[2k], blue formation draws[2k+1]
        pairs, matches = [], []
        root = stream_seed(NS, "match", SEED_INDEX, period) % (2**62)
        for k in range(games):
            r_bot, b_bot = cells[k % len(cells)]
            rc, bc = canon[int(draws[2 * k])], canon[int(draws[2 * k + 1])]
            pairs.append(SetupPair(setup_pair_id=k, red_setup=to_engine_setup(rc, RED), blue_setup=to_engine_setup(bc, BLUE),
                                   generation_seed=0, bank_version=BANK_VERSION, generation_family="library_uniform_with_replacement"))
            matches.append(MatchSpec(candidate=policy_ref(r_bot), opponent=policy_ref(b_bot), setup_pair_id=k, candidate_color=RED,
                                     replicate=0, root_seed=root, setup_bank_version=BANK_VERSION, rules=COLLECTOR_RULES))
        bank = SetupBank(bank_version=BANK_VERSION, root_seed=root, generation_family="library_uniform_with_replacement", pairs=tuple(pairs))
        for m in matches: m.resolve_setups(bank)
        t1 = time.perf_counter()
        run = run_schedule(matches, bank, worker_count=WORKERS, record_actions=False)
        t_play = time.perf_counter() - t1
        if run.policy_errors: raise RuntimeError(f"period {period}: {run.policy_errors} policy errors")
        outcomes = {"red_win": 0, "blue_win": 0, "draw": 0}
        for r in run.results:
            k = int(r.setup_pair_id)
            if r.draw: z_red = 0; outcomes["draw"] += 1
            elif r.winner == RED: z_red = 1; outcomes["red_win"] += 1
            else: z_red = -1; outcomes["blue_win"] += 1
            buffer.add_outcome(fps[int(draws[2 * k])], z_red)
            buffer.add_outcome(fps[int(draws[2 * k + 1])], -z_red)
        t2 = time.perf_counter()
        result = trainer.update(buffer, global_iteration=period)
        t_upd = time.perf_counter() - t2
        terms = dict(result.epoch_terms[-1]) if result.epoch_terms else {}
        rec = {"period": period, "distinct_formations": len(distinct), "reflected": n_refl, "games": len(run.results),
               "outcomes": outcomes, "ready_rows": result.ready_rows, "optimizer_steps": result.optimizer_steps,
               "alpha": result.alpha, "clip_activations": result.clip_activations,
               "policy_loss": terms.get("policy_loss"), "value_loss": terms.get("value_loss"), "behavior_kl": terms.get("behavior_kl"),
               "advantage_std": terms.get("advantage_std"), "mean_prefix_entropy_nats": terms.get("mean_prefix_entropy_nats"),
               "raw_digest": result.digest_after, "ema_digest": result.ema_digest_after,
               "seconds": {"score": round(t_score, 1), "play": round(t_play, 1), "update": round(t_upd, 1), "total": round(time.perf_counter() - t0, 1)}}
        if not a.smoke:
            with open(OUT / "periods.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
        log(f"p{period:03d} games {len(run.results)} rw/bw/d {outcomes['red_win']}/{outcomes['blue_win']}/{outcomes['draw']} | rows {result.ready_rows} steps {result.optimizer_steps} | "
            f"pol {terms.get('policy_loss', float('nan')):.4f} val {terms.get('value_loss', float('nan')):.4f} kl {terms.get('behavior_kl', float('nan')):.5f} H {terms.get('mean_prefix_entropy_nats', float('nan')):.3f} | "
            f"ema {result.ema_digest_after[:8]} | {rec['seconds']['total']:.0f}s (play {t_play:.0f})")
        if not a.smoke and (period % CHECKPOINT_EVERY == 0 or period == periods):
            trainer.save_checkpoint(OUT / f"ckpt_{period:04d}")
            state_path.write_text(json.dumps({"period": period, "raw": result.digest_after, "ema": result.ema_digest_after}) + "\n")
            log(f"checkpoint ckpt_{period:04d}")
    log("TRAINING DONE" if not a.smoke else "SMOKE OK")

if __name__ == "__main__":
    main()
