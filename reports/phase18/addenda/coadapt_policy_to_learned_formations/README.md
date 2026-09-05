# Operator addendum v4 — letting the neural policy CO-ADAPT to the learned formations

**Informal, operator-requested. Not Gate G3 evidence.** Run 2026-09-05 from the
`phase18/g3-stage6b-harness` tree; runtime under `output/phase18/runtime/addendum_coadapt/`
(ignored; bundles and game chunks not committed).

## Question

Addendum v3 showed the from-scratch setup model's period-1,024 formations give the
*unchanged* G3 control policy +13 points. That policy had never trained on such
formations. Does training the policy on them — co-adaptation — add more?

## Design: a matched pair with G3's own control lineage

A G3 control-style lineage (setup model **frozen**, C1 trained on the canonical/live
mixture) run through the production `LineageRunner` untouched, with exactly one
redirection: the frozen setup model is the from-scratch model's period-1,024 EMA
(`c0f532e75d80…`) instead of G3's random init (`082ff778…`). Everything else is G3's
control lineage — the same C1 initialisation (`cfe60bb0cb34…`, the canonical Phase 8
seed), the same collector and C1 seeds (G3's namespace), the same per-period budget,
the same bundle cadence. The control-freeze assertion held every period against the
ckpt_1024 digest; zero setup updates; every integrity counter zero.

The lineage was killed at period 251 by a host out-of-memory crash (the second such
crash on a G3-style leg; both struck 7–10 h in). Rather than replay to 256, the
operator chose to evaluate the **period-224 bundle** (C1 at 14,336 of 16,384 updates).
That is budget-matched against G3's own control at *its* period-224 bundle, and the
224→256 budget effect was measured directly and is null (below). Periods 225–251 are
archived under `runtime/control/archive/`, not deleted; the replay can resume from
`bundle_0224` at any time.

Evaluation as in addendum v3: the frozen 2,560-game G3 schedule, G3 seeds, greedy,
float32, battleless 200; own formation from `ckpt_1024` / library / G3-init; the G3
paired family-stratified bootstrap. Four new neural arms, 10,240 games, zero errors.

## Result

### The same formations, two policies (both at 14,336 updates)

| mover | trained on | EWR with ckpt_1024 formations |
|---|---|---|
| G3 control C1 @224 | random-init formations | 0.7309 |
| **co-adapted C1 @224** | **ckpt_1024 formations** | **0.7512** |

**Primary, budget-matched, paired: +0.0203 [+0.0024, +0.0380].** Real, and modest.

### Every mover × formation source

| mover | ckpt_1024 | library | G3 init |
|---|---|---|---|
| co-adapted C1 @224 | 0.7512 | 0.7045 | 0.6049 |
| G3 control C1 @224 | 0.7309 | — | — |
| G3 control C1 @256 (v3) | 0.7324 | 0.6648 | 0.6025 |

| contrast | difference (95%) |
|---|---|
| co-adapted − control, **ckpt_1024** formations (budget-matched) | **+0.0203 [+0.0024, +0.0380]** |
| co-adapted − control, **library** formations | **+0.0396 [+0.0207, +0.0586]** |
| co-adapted − control, **G3-init** formations | +0.0023 [-0.0149, +0.0196] |
| co-adapted @224 − control @256, ckpt_1024 formations | +0.0187 [+0.0019, +0.0358] |
| control @224 − control @256, ckpt_1024 formations (budget effect) | -0.0016 [-0.0197, +0.0168] |
| co-adapted C1: ckpt_1024 − library formations | +0.0467 [+0.0172, +0.0774] |

### Primary contrast by opponent

| opponent | difference |
|---|---|
| stress_berserker | +0.0375 |
| strategic_rule_based | +0.0344 |
| stress_scout_rush | +0.0344 |
| stress_chaos | +0.0203 |
| stress_miner_rush | +0.0172 |
| tactical_rule_based | +0.0141 |
| stress_information_miser | +0.0078 |
| basic_heuristic | -0.0031 |

## Reading

1. **Co-adaptation adds about +2 points on top of the formations' +13.** Roughly 85%
   of the total benefit comes from *using* good formations; about 15% from training the
   policy on them. The setup model's value to the neural player is mostly drop-in.
2. **It does not narrow the policy.** The co-adapted C1 is *better* with handcrafted
   library formations too (+4.0), and no worse with the poor G3-init formations (+0.2,
   null). Training on good formations taught it something general about playing from
   well-built boards, not a specialisation to one generator.
3. **The last 32 periods of a G3 leg do nothing measurable** (budget effect −0.002),
   which is why the period-224 comparison stands in for 256.
4. The whole arc, in one line: G3's FAIL was a setup model that never learned; a setup
   model that does learn is worth +13 to the policy as a drop-in and +15 with
   co-adaptation, against a +5 gate.

## Caveats

One seed; informal; no pre-registered rule. Period 224, not 256, for both movers. The
library arm's formations come from the same evaluation library as the opponents'.
