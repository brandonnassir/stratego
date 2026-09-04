# Phase 18 Gate G3 — result: FAIL

Run `G3-PILOT-2026-A`. Evidence created at `8c1baa8`; analysed with the repaired
persisted-receipt reader at `ffb799e` under the analysis-only rebind (P18-A002).

## Decision

**FAIL** — the frozen rule does not pass.

> PROCEED requires the 95% lower bound above zero **and** the point estimate at least 0.05.

| | |
|---|---|
| contrast | `EWR(candidate_final) − EWR(control_final)`, paired by case |
| candidate EWR | 0.605469 |
| control EWR | 0.602539 |
| **point estimate** | **+0.002930** |
| **95% interval** | **[−0.017040, +0.022694]** |
| margin | 0.050 |
| lower bound above zero | no |
| point at margin | no |
| near boundary | no |

Stratified cluster bootstrap over bases within families: 2,560 cases, 160 bases,
16 families, 10,000 replicates, seed `2617613076104066311`. Bootstrap standard
error 0.010242, closely matching the direct per-base standard error 0.010333.

This is not a near-miss. The margin of +0.05 sits far outside the interval, whose
upper end (+0.0227) is less than half the margin. The design's near-boundary
branch — the one that would justify a conditional second seed — does **not**
apply, and no second seed was launched.

The result is informative rather than merely null: enabling setup learning
together with the policy co-adaptation it causes moved play by +0.3 percentage
points of expected win rate, and an effect as large as the +0.05 the gate asked
for is excluded at 95% confidence.

## The decision rests on a clean comparison

All nine computable gates pass and every fairness condition holds, so `FAIL` is a
real result rather than a blocked one.

| gate | |
|---|---|
| G1 legality | pass |
| G2 orientation | pass |
| G3 attribution | pass |
| G4 accounting | pass |
| G5 bundle identity | pass |
| G6 restart | pass |
| G7 paired | pass |
| G8 diversity | pass |
| G9 finite and valid | pass |
| G10 clean deliverable | judged at review, not computed here |

`failed_gates: []`, `fairness_all_hold: true`, `fairness_problems: []`.

Arms: candidate `2162b448017ca844` and control `e1c410021f71124f`, both at period
256, each played over the frozen 2,560-case schedule. The two arms chose a
different own setup in 864 of the 2,560 cases; on the rest the learned setup
model reproduced the frozen one's choice.

## Breakdown

By opponent (mean paired difference, n = 320 each):

| opponent | difference |
|---|---|
| strategic_rule_based | +0.0406 |
| tactical_rule_based | +0.0297 |
| stress_chaos | +0.0047 |
| stress_scout_rush | +0.0047 |
| stress_information_miser | −0.0047 |
| basic_heuristic | −0.0078 |
| stress_miner_rush | −0.0156 |
| stress_berserker | −0.0281 |

The two gains are against the rule-based opponents; the stress opponents are flat
or slightly negative. By colour the effect is symmetric (+0.0023 and +0.0035).
Across the 16 families the per-family differences run from −0.0719 (F05) to
+0.0938 (F11) with a per-base standard deviation of 0.1326 — spread consistent
with noise at this sample size, not a stable subgroup effect. Nothing here is a
pre-registered subgroup hypothesis, and none of it changes the primary decision.

## Scope

The estimand is the **total** benefit of enabling setup learning plus the policy
co-adaptation it causes. The setup network's isolated causal contribution is not
identified by this design, so this result does not say the setup model learned
nothing — only that the whole intervention did not move play against these eight
handcrafted opponents by the margin the gate required.

## Provenance

The primary contrast was computed exactly once, after the amendment, the reader
repair, the regression tests and the rebind record were all committed and pushed.
No alternative calculation, diagnostic arm or second seed was run.
