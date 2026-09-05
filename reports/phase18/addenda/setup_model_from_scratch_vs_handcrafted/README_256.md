# Operator addendum v2 — a setup model trained FROM SCRATCH on handcrafted games only

**Informal, operator-requested. Not Gate G3 evidence.** Run 2026-09-04 from the
`phase18/g3-stage6b-harness` tree; runtime (checkpoints, 143k evaluation rows) under
`output/phase18/runtime/addendum_library_setup_from_scratch/` (ignored).

## Question

Does a setup model that learns *only* by watching the handcrafted bots play each other —
with formations drawn from the handcrafted library — make those bots better? No neural
policy anywhere; the movers are the fixed rule-based bots throughout.

## Design

- **Fresh model**, new init (`c549bc02…`, independent of G3's `082ff778…`), the G3 setup
  architecture and the G3 setup trainer **byte-for-byte** (PPO-clip policy loss, value
  head, entropy-prediction head, behaviour KL, alpha schedule, EMA 0.999, lr 5e-5,
  5 epochs/update, batch 1,024).
- **Training data**: each period, 2,048 games; game *k* is cell *k* mod 56 of the 56
  ordered pairs of distinct handcrafted bots (no self-play); each side's formation is
  drawn **uniformly with replacement** from the 6,400 training-split library formations
  (indices 0..399, all 16 families). Formations enter the buffer as teacher-forced rows
  scored under the current raw model; outcomes (+1/0/−1 per side) attribute to the
  formation used. The model never plays and never generates a training formation.
- 256 periods, **524,288 games**, ~3,024 distinct formations observed per period,
  1.76 h. Checkpoints every 32 periods.
- **Evaluation**: identical to addendum v1. Each of the 8 bots plays the frozen G3
  schedule (160 evaluation formations × 8 opponents × 2 colours = 2,560 paired games)
  with its own formation sampled from a checkpoint's EMA model (G3 evaluation seeds),
  from the init model, or a same-family library formation. G3 bootstrap, unchanged.
  Evaluation formations (indices 400..409) were never seen in training.

## Result

### Learning curve — pooled over 8 students, each checkpoint vs the init (`ckpt_0`)

| checkpoint | EWR | vs init, 95% |
|---|---|---|
| ckpt_0 (init) | 0.3487 | — |
| ckpt_32 | 0.3489 | +0.0003 [-0.0048, +0.0054] |
| ckpt_64 | 0.3501 | +0.0014 [-0.0054, +0.0084] |
| ckpt_128 | 0.3689 | +0.0202 [+0.0120, +0.0286] |
| ckpt_192 | 0.3786 | +0.0299 [+0.0219, +0.0382] |
| ckpt_256 | 0.3955 | +0.0469 [+0.0379, +0.0560] |
| library | 0.4989 | ckpt_0 − library -0.1503 [-0.1704, -0.1293] |

Nothing for the first 64 periods, then a steady, **accelerating** climb: +2.0 points by
128, +3.0 by 192, **+4.7 by 256** — and the curve had not flattened. The policy entropy
fell from 1.84 to 1.68 nats/prefix over the run (G3's trainer never left 1.81).

### Per student (EWR over 2,560 games)

| student | init | ckpt_256 | library | 256 − init | 256 − library |
|---|---|---|---|---|---|
| basic_heuristic | 0.4344 | 0.4820 | 0.5820 | **+0.0477** | -0.1000 |
| strategic_rule_based | 0.5021 | 0.5742 | 0.7498 | **+0.0721** | -0.1756 |
| stress_berserker | 0.2898 | 0.3342 | 0.4057 | **+0.0443** | -0.0715 |
| stress_chaos | 0.2203 | 0.2574 | 0.3434 | **+0.0371** | -0.0859 |
| stress_information_miser | 0.2283 | 0.2584 | 0.3223 | **+0.0301** | -0.0639 |
| stress_miner_rush | 0.3154 | 0.3588 | 0.4477 | **+0.0434** | -0.0889 |
| stress_scout_rush | 0.2928 | 0.3271 | 0.4139 | **+0.0344** | -0.0867 |
| tactical_rule_based | 0.5061 | 0.5721 | 0.7268 | **+0.0660** | -0.1547 |

**All eight** students improve, every one with its own 95% interval excluding zero
(+3.0 to +7.2 points). And **all eight** remain well below the library formation.

### Gain by opponent (ckpt_256 − init, pooled)

| opponent | difference |
|---|---|
| stress_chaos | +0.0760 |
| basic_heuristic | +0.0619 |
| stress_berserker | +0.0613 |
| stress_scout_rush | +0.0529 |
| strategic_rule_based | +0.0482 |
| tactical_rule_based | +0.0451 |
| stress_miner_rush | +0.0291 |
| stress_information_miser | +0.0004 |

## Reading, next to the G3 model (addendum v1)

| formation source | pooled EWR | vs its init | vs library |
|---|---|---|---|
| G3 candidate init (`082ff778`) | 0.4494 | — | -0.0495 |
| G3 candidate after 256 periods (self-sampled formations) | 0.4589 | +0.0095 | -0.0400 |
| this init (`c549bc02`) | 0.3487 | — | -0.1503 |
| this model after 256 periods (library formations, handcrafted games) | 0.3955 | **+0.0469** | -0.1034 |
| handcrafted library | 0.4989 | | |

1. **Yes — a setup model trained only on handcrafted games makes every handcrafted bot
   better.** +4.7 points pooled, all eight bots individually significant, still rising.
   That is roughly 5× the +0.95 the G3 model transferred, for the same number of
   periods and outcomes.
2. **Learning from the library's games is a far stronger signal than learning from the
   model's own samples.** G3's trainer sat at maximum entropy for 256 periods; this one
   concentrated. Same loss, same recipe — the difference is what it watched.
3. **The model family still starts far below the handcrafted library.** This init was
   15 points under it (G3's happened to be 5 under — the init lottery is large), and 256
   periods closed about a third of that gap. Its formations are better than an untrained
   model's, and worse than a human-designed one.
4. The steep, unflattened curve says the honest next question is horizon: this run
   was stopped at G3's budget, not at convergence.

## Caveats

One seed; informal; no pre-registered rule. The library arm's formations come from the
same evaluation library as the opponents' (a different formation of the same family).
Model arms sample one formation per case; the library arm uses a fixed one.
