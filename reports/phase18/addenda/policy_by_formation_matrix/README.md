# Operator addendum v7 — the policy × formation matrix, with the untrained and Phase 8 baselines

**Informal, operator-requested. Not Gate G3 evidence.** Run 2026-09-06. All numbers are EWR
on the frozen G3 exam: 2,560 paired games (160 library opponent formations × 8 handcrafted
opponents × 2 colours), battleless 200, greedy, float32, G3 seeds. "Formations" is where the
neural policy's *own* starting board comes from; the opponents always play library boards.

## New measurements in this addendum

| policy | training | formations | EWR |
|---|---|---|---|
| untrained C1 + untrained setup model (G3 control `bundle_0000`) | 0 updates | G3 init setup `082ff778` | **0.2770** |
| Phase 8 warm-start C1 (`checkpoints/phase8/warmstart_c1_v1.pt`, eval export `agent07/warmstart_eval.pt`) | 24,000 updates, fixed 28,000-game corpus only | library (same family, rotated) | **0.6889** |
| Phase 8 warm-start C1 | 24,000 updates | pretrained setup model `ckpt_1024` (addendum v2) | **0.7248** |
| Phase 8 warm-start C1 | 24,000 updates | G3 init setup `082ff778` (what `control_final` played) | **0.5891** |

The Phase 8 model's untrained initialisation has checksum `cfe60bb0…`, identical to the C1
init every Phase 18 lineage started from, so every policy below shares one starting point.

## The full matrix (all addenda)

| policy | updates | training diet | G3-init formations | library formations | pretrained (ckpt_1024) formations |
|---|---|---|---|---|---|
| untrained | 0 | — | 0.2770 | — | — |
| Phase 8 warm-start | 24,000 | corpus only | 0.5891 | 0.6889 | 0.7248 |
| G3 control | 16,384 | 50/50 corpus + live games on G3-init formations | 0.6025 | 0.6648 | 0.7324 |
| G3 candidate (its own learned formations `ea1a809b`) | 16,384 | 50/50, formations learning from own samples | 0.6055 (own) | — | — |
| v4 co-adapted | 14,336 | 50/50 corpus + live games on ckpt_1024 formations, setup frozen | 0.6049 | 0.7045 | **0.7512** |

## Reading

1. **From 0.277 to 0.751.** The untrained policy wins 28% of the exam; the best pipeline
   (pretrained setup model + a policy trained on its formations) wins 75%.
2. **The pretrained formations are portable but not equally exploitable.** Handed to the
   Phase 8 model they add +3.6; to the G3 control, +6.8; the policy trained *on* them
   reaches 0.751. A policy exploits model-generated formations better the more it has
   trained on model-generated formations.
3. **Diet alone does not beat Phase 8.** The G3 control (mixed diet, no good formations)
   scores 0.665 with library formations, below Phase 8's 0.689. What lifts a policy above
   Phase 8 is the pretrained setup model — as a drop-in (+3.6 for Phase 8 itself) and far
   more through co-adaptation (0.705 with library formations, 0.751 with the pretrained
   ones, with 40% fewer updates than Phase 8).
4. **With the same poor formations, updates are not the story**: Phase 8 (24k) 0.589 vs the
   G3 control (16k) 0.603.

## Caveats

Informal; one seed; the v4 policy sits at 14,336 updates against Phase 8's 24,000, so a
matched-budget co-adaptation run remains the clean confirmation. Phase 8's own published
numbers were on a different protocol (vs random, Phase 4 rules) and are not comparable to
this exam.
