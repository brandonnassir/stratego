# Operator addendum v8 — matched budget: pretrained setup + co-adapted policy (24,000 updates) vs the Phase 8 model (24,000 updates)

**Informal, operator-requested. Not Gate G3 evidence.** Run 2026-09-06/07 from the
`phase18/g3-stage6b-harness` tree; runtime under `output/phase18/runtime/addendum_coadapt/`
(ignored; bundles and game chunks not committed).

## Question

Addendum v4's co-adapted policy (14,336 updates) beat the Phase 8 warm-start model
(24,000 updates). Does that hold at **equal budget**? The operator's hope: a pretrained
setup model plus a policy co-adapted to it, trained for Phase 8's budget, exceeds Phase 8.

## Design

The v4 lineage — G3 control-style, setup model **frozen at the from-scratch model's
period-1,024 EMA** (`c0f532e7…`), C1 trained from the same untrained init Phase 8 used
(`cfe60bb0…`) on the 50/50 corpus + live-game mixture — extended from its verified
`bundle_0224` to **375 periods = 24,000 C1 updates**, Phase 8's budget exactly.

**Recorded deviation.** The runner's bundle identity includes the horizon, so a
375-period config refuses bundles written under the 256-period one. The source bundle
(`bundle_0224`) was verified under its own 256-period identity for the resume only, then
the true 375-period identity was restored; every bundle from 256 onward carries it. The
C1 learning-rate schedule is linear warm-up (500 steps) then constant, so the horizon
does not enter the training dynamics; the only difference from an uninterrupted 375-period
run is MPS non-determinism, which applies to every G3-style leg. One watchdog restart
(a swap-cap false alarm at 39% free memory) replayed periods 321–348 from `bundle_0320`;
all archived, nothing deleted.

Evaluation as in v3/v4: the frozen 2,560-game G3 schedule, G3 seeds; own formation from
the pretrained setup model (`ckpt_1024`), a same-family library formation, or G3's init
setup model; paired case-by-case against the Phase 8 model's rows on the same three
sources (addendum v7), the G3 control's (v3), and v4's own period-224 rows.

## Result

### The same formations, four policies

| policy | updates | ckpt_1024 | library | G3 init |
|---|---|---|---|---|
| **co-adapted, 24k (this run)** | 24,000 | **0.7395** | **0.7066** | **0.6158** |
| Phase 8 warm-start | 24,000 | 0.7248 | 0.6889 | 0.5891 |
| co-adapted, 14k (v4, period 224) | 14,336 | 0.7512 | 0.7045 | 0.6049 |
| G3 control | 16,384 | 0.7324 | 0.6648 | 0.6025 |

### Paired contrasts (95%)

| contrast | difference |
|---|---|
| **matched budget, pretrained formations**: co-adapted 24k − Phase 8 | **+0.0146 [-0.0029, +0.0317]** |
| matched budget, library formations | +0.0178 [-0.0020, +0.0379] |
| matched budget, G3-init formations | +0.0268 [+0.0072, +0.0469] |
| 224 → 375 periods, pretrained formations | -0.0117 [-0.0290, +0.0058] |
| 224 → 375 periods, library formations | +0.0021 [-0.0156, +0.0203] |
| 224 → 375 periods, G3-init formations | +0.0109 [-0.0076, +0.0291] |
| co-adapted 24k: pretrained − library formations | +0.0328 [+0.0017, +0.0645] |

### Matched budget, pretrained formations, by opponent

| opponent | difference |
|---|---|
| stress_scout_rush | +0.0266 |
| strategic_rule_based | +0.0250 |
| stress_miner_rush | +0.0219 |
| stress_chaos | +0.0125 |
| tactical_rule_based | +0.0125 |
| stress_information_miser | +0.0094 |
| basic_heuristic | +0.0063 |
| stress_berserker | +0.0031 |

## Reading

1. **At equal budget the co-adapted pipeline beats Phase 8 on every formation source and
   against all eight opponents — by +1.5 to +2.7 points.** Only the G3-init contrast is
   individually significant; the other two intervals just touch zero. The hope holds
   directionally; the margin is modest.
2. **The extra 9,700 updates bought nothing.** 224 → 375 periods moved the pretrained-
   formation score by −1.2 (within noise), library +0.2, G3-init +1.1: the co-adapted
   policy had plateaued by ~14,000 updates. v4's under-budget lead over Phase 8 was
   therefore a plateaued policy against a plateaued policy, not a budget artefact.
3. **The formations remain the lever.** Swapping Phase 8's formations for the pretrained
   ones is worth +3.6 with no retraining; the best policy-training recipe found here is
   worth about +2 over Phase 8 at equal budget. Against these opponents the policy's
   ceiling with the pretrained formations sits near 0.74–0.75 whichever way it is trained.
4. **Why not more?** Both training recipes saturate well inside 24,000 updates, and the
   exam is the same eight handcrafted bots that bound the setup model (v5, v6). Further
   gains would have to come from a different opponent pool or signal, not from more of
   this training.

## Caveats

One seed; informal; the horizon-extension deviation above; the evaluation and Phase 8
comparison share the operator-chosen library-formation rotation. Phase 8's own published
numbers were on a different protocol and are not comparable.
