# Operator addendum v6 — training the setup model on half the handcrafted roster

**Informal, operator-requested. Not Gate G3 evidence. PAUSED after half A by operator
decision; half B not run.** Run 2026-09-06 from the `phase18/g3-stage6b-harness` tree;
runtime under `output/phase18/runtime/addendum_half_roster/` (ignored). Half B can be
resumed later by relaunching `chain.sh` — its directory holds only the reusable library
rows.

## Question

Addendum v5 showed a 2× setup model reaches the same ceiling as the 1× on identical
games, pointing at the training data rather than capacity. Is the *diversity* of the
eight handcrafted opponents the binding signal? Test: train the same model on only four
of them and see whether the ceiling drops, and whether it drops specifically against
the four it never saw.

## Design

The 1× from-scratch trainer with the roster restricted to **half A = basic_heuristic, stress_information_miser, stress_miner_rush, tactical_rule_based**
(12 ordered pairs instead of 56). Everything else identical to the full-roster run: the
same init (`c549bc02…`), the same 2,048 games per period (so the *amount* of signal is
matched and only its diversity changes), the same library draws and match seeds, the
same recipe, 1,024 periods. Half B (strategic_rule_based, stress_berserker, stress_chaos, stress_scout_rush) was pre-specified
by alternating the roster order but not run.

Evaluation as in v2/v5: each of the 8 bots plays the frozen 2,560-game G3 schedule with
its own formation from each checkpoint's EMA or a same-family library formation; the G3
paired bootstrap, unchanged. Results are split by whether the *opponent* was in the
training roster.

## Result

### Absolute EWR by checkpoint, pooled over 8 handcrafted movers

| period | full roster (8) | half A (4) | half − full |
|---|---|---|---|
| 0 | 0.3487 | 0.3487 | +0.0000 |
| 256 | 0.3955 | 0.3834 | -0.0121 |
| 384 | 0.4383 | 0.4246 | -0.0137 |
| 512 | 0.4717 | 0.4563 | -0.0154 |
| 640 | 0.4959 | 0.4793 | -0.0166 |
| 768 | 0.5112 | 0.5056 | -0.0056 |
| 896 | 0.5312 | 0.5209 | -0.0104 |
| 1024 | 0.5359 | 0.5305 | -0.0054 |
| library | 0.4989 | | |

At 1,024: full +0.0370 [+0.0169, +0.0576] vs library; half A
+0.0315 [+0.0116, +0.0522]. The gap opened to −1.7 points by
period 640 and then **closed to −0.5** as both plateaued.

### By opponent, ckpt_1024 − library

| opponent | in half A's roster | full model | half-A model | half − full |
|---|---|---|---|---|
| basic_heuristic | seen | +0.0309 | +0.0482 | +0.0174 |
| strategic_rule_based | unseen | +0.0582 | +0.0412 | -0.0170 |
| stress_berserker | unseen | +0.0844 | +0.0732 | -0.0111 |
| stress_chaos | unseen | +0.0441 | +0.0266 | -0.0176 |
| stress_information_miser | seen | +0.0150 | +0.0090 | -0.0061 |
| stress_miner_rush | seen | -0.0008 | -0.0006 | +0.0002 |
| stress_scout_rush | unseen | +0.0111 | +0.0094 | -0.0018 |
| tactical_rule_based | seen | +0.0527 | +0.0453 | -0.0074 |

Group means (model − library): on the four bots half A trained against, full
+0.0245 vs half A +0.0255 — **identical**; on the four it never saw,
full +0.0495 vs half A +0.0376 — half A is **~1.2 points behind**,
but still clearly above the library on bots it never watched.

## Reading

1. **Halving the roster barely moved the ceiling.** −0.5 points pooled at 1,024, inside
   the noise of a single checkpoint, and both models beat the handcrafted library by
   3–4 points.
2. **The small cost is where it should be — on unseen opponents** (−1.2), with none on
   the seen ones. So roster diversity does buy generalisation, but the four-bot model
   still learned formations that beat the library against all eight.
3. **Together with v5, the ceiling is now known to be insensitive to both model size and
   roster size.** What sets it looks like the *kind* of signal — rule-based bots playing
   library formations under this recipe — rather than how many bots or how big a model.
   Moving it would mean qualitatively different opponents (the neural policy, self-play
   against the setup model's own formations) or a different learning signal, not more of
   the same.

## Caveats

One seed; informal; half B not run, so the result could depend on which four bots were
chosen (half A holds one strong rule-based bot and three weaker ones). "Seen/unseen" is by
opponent; every bot is also a mover in the exam.
