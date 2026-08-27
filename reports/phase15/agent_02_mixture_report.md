# Phase 15 — Agent 2 follow-up
## Belief-mixture pilot: does `lambda*B24 + (1-lambda)*count` rescue deeper search?

**Question.** The deeper-search pilot found LARGE and XLARGE both *worse* than MEDIUM (-0.075 paired EWR each) while the oracle at the same budgets got *better* (+0.042 at LARGE) — the world distribution, not the search mechanics, is what fails at scale. Can mixing B24's marginals with the robust remaining-count marginals fix the distribution?

**Answer: no useful mixture. The experiment closes at Stage 1 and MEDIUM + B24 stands.**

Nothing was trained and nothing in the search changed. The P24 weights, the B24 specialist and its applied temperature, the candidate rule, `beta`, `epsilon`, world deduplication, the legal-world sampler and every per-decision seed are the frozen objects, reached by import. The pilot adds exactly one belief provider.

## 1. What was varied

One vector: the 12-way rank marginal handed to the accepted sampler, replaced by `normalize(lambda * b_B24 + (1 - lambda) * b_count)`. The gate checks that this is literally what happens, on real public states, at every lambda.

| preset | worlds | depth | candidates | beta | epsilon | dedup |
|---|---|---|---|---|---|---|
| LARGE | 64 | 9 | 8 | 0.1 | 1e-06 | yes |
| MEDIUM | 32 | 8 | 8 | 0.1 | 1e-06 | yes |

Gate checks, all on the frozen bytes:

| check | result | detail |
|---|---|---|
| configuration_invariants | PASS | - |
| determinism_and_legality | PASS | - |
| frozen_identity | PASS | - |
| mixture_algebra | PASS | - |
| worlds_legal | PASS | - |

**One thing the two endpoint names hide.** `lambda = 1.00` really is B24: it reproduces the frozen `p24_b24` arm's chosen action at 120 of 120 positions, which is a check of the mixture code path rather than a tautology. `lambda = 0.00` is **not** the accepted `remaining_count` provider: that one draws from the count-uniform skeleton (`weight = remaining_count`), while a mixture at zero feeds count marginals to the *learned* sampler, whose weight is `learned_probability * remaining_count` — an effective `count^2`. Keeping one sampler across the sweep is what makes the sweep a measurement of lambda, so the accepted baseline is carried as its own separate arm and reported separately.

## 2. Stage 1 — the position diagnostic

120 clean replayed positions, 1080 decisions, seed `20260824` for every arm at every position, LARGE search only. No games were played.

The reference is the **oracle at the same rung**. Root candidates come from P24's policy alone, which no belief provider can influence, so every arm and the oracle evaluate the *same* candidate set at every position — the shared-candidate check below confirms it on all 1080 decisions. That makes

```text
oracle_q_regret(arm) = max_a Q_oracle(a) - Q_oracle(a chosen by arm)
```

well defined everywhere, and it is a finer instrument than agreement: two arms can miss the oracle's move equally often while one of them loses far less by doing so.

**Read the regret column against a floor of 0.0458, not against zero.** The oracle selects by `S = Q + beta*log(pi + epsilon)`, not by `argmax Q`, so it gives up some true-world value itself — at 79 of 120 positions its own choice is not the Q-maximizing candidate. `beta` is frozen and identical across every arm, so the floor is common and the honest quantity is the *excess* over it, shown as its own column below.

| arm | oracle agreement | oracle Q-regret | excess over floor | median | disagrees with B24@MEDIUM | disagrees with B24@LARGE | illegal | search errors |
|---|---|---|---|---|---|---|---|---|
| oracle @ LARGE (ceiling, offline) | 1.000 | 0.0458 | 0.0000 | 0.0163 | - | - | 0 | 0 |
| B24 @ MEDIUM (incumbent) | 0.858 | 0.0628 | 0.0170 | 0.0311 | 0.000 | - | 0 | 0 |
| B24 @ LARGE (the regression) | 0.908 | 0.0636 | 0.0178 | 0.0345 | 0.092 | 0.000 | 0 | 0 |
| remaining_count @ LARGE (accepted baseline) | 0.908 | 0.0571 | 0.0113 | 0.0363 | 0.100 | 0.017 | 0 | 0 |
| mix lambda=0.00 @ LARGE | 0.908 | 0.0588 | 0.0130 | 0.0411 | 0.117 | 0.033 | 0 | 0 |
| mix lambda=0.25 @ LARGE | 0.917 | 0.0571 | 0.0113 | 0.0363 | 0.108 | 0.025 | 0 | 0 |
| mix lambda=0.50 @ LARGE | 0.908 | 0.0638 | 0.0180 | 0.0363 | 0.100 | 0.033 | 0 | 0 |
| mix lambda=0.75 @ LARGE | 0.917 | 0.0627 | 0.0169 | 0.0311 | 0.083 | 0.025 | 0 | 0 |
| mix lambda=1.00 @ LARGE | 0.908 | 0.0636 | 0.0178 | 0.0345 | 0.092 | 0.000 | 0 | 0 |

Latency is omitted from this table on purpose: Stage 1 ran ten single-threaded workers at once, so its ~6.8 s/move at LARGE is a contention figure, not a shippability figure. The deeper-search pilot's *idle* measurement (3.82 s median, 3.92 s p95 at LARGE) is the one that means anything, and nothing here changes it — the mixture adds one vector addition per decision.

### The measurement that decides this

B24 at MEDIUM scores **0.9333** in match play; B24 at LARGE scores **0.8583**. That 0.075 gap is the entire reason this pilot exists. Both arms are in the table above. Their paired oracle Q-regret differs by

```text
b24@LARGE - b24@MEDIUM  =  +0.00082  ± 0.00167   (5 positions better, 5 worse, 110 tied of 120)
```

**The metric cannot see the gap.** The arm that wins the games and the arm that loses them are indistinguishable on position-level regret — half a standard error apart, tied at 110 of 120 positions. That is not a defect in the instrument; it is the result. The deeper rung's regression does not live in the quality of individual root decisions, so no reweighting of the marginals those decisions are made from can address it. Every lambda below is therefore being asked to fix something it has no contact with, and the flat, non-monotone sweep is exactly what that looks like.

Legality and fallback: 0 illegal decisions and 0 search errors across every arm. The mixture introduces no correctness problem at any lambda; whatever the sweep shows, it is not a broken provider.

## 3. Which lambda, if any

The rule, written down before the numbers: *select an interior lambda only if its paired oracle Q-regret beats both endpoints by more than the paired standard error of that difference; otherwise report no useful mixture*.

| interior lambda | oracle Q-regret | paired vs lambda=0 (neg = better) | paired vs lambda=1 (neg = better) | positions better than lambda=1 | worse |
|---|---|---|---|---|---|
| mix lambda=0.25 @ LARGE | 0.0571 | -0.0018 ± 0.0018 | -0.0066 ± 0.0072 | 2 | 1 |
| mix lambda=0.50 @ LARGE | 0.0638 | 0.0050 ± 0.0074 | 0.0002 ± 0.0007 | 2 | 2 |
| mix lambda=0.75 @ LARGE | 0.0627 | 0.0039 ± 0.0075 | -0.0009 ± 0.0009 | 2 | 0 |

**No interior lambda clears the rule.** The findings, arm by arm, in the order the rule considered them:

- mix_l025_LARGE: regret vs count endpoint -0.0018 (se 0.0018), vs B24 endpoint -0.0066 (se 0.0072)
- mix_l075_LARGE: regret vs count endpoint +0.0039 (se 0.0075), vs B24 endpoint -0.0009 (se 0.0009)
- mix_l050_LARGE: regret vs count endpoint +0.0050 (se 0.0074), vs B24 endpoint +0.0002 (se 0.0007)

Stage 2 is therefore not run. Playing 30-40 games to confirm a difference the 120-position diagnostic cannot resolve would be spending hours to add noise to a null result, and the brief says so explicitly: *if none beats pure B24 or count convincingly, stop and report no useful mixture*.

## 4. What this does and does not say

- The mixture is a *decision-time* change only. It cannot fix a marginal that is wrong; it can only blend it toward a prior that is coarse but never confidently wrong.
- **Belief quality was never the binding constraint here, and the deeper-search pilot's own numbers say so.** On that 60-board pack a *perfect* belief scored 0.8833 at MEDIUM and 0.9250 at LARGE, while B24 scored 0.9333 at MEDIUM. The ceiling the mixture is reaching toward sits at or below the incumbent. Blending toward a coarser prior cannot beat a target that is already behind you.
- What the oracle *does* show is a direction: it is the only arm whose score rises with depth (+0.0417 at LARGE, +0.0250 at XLARGE, paired against its own MEDIUM). Deeper search pays only when the worlds are right, and a fixed convex blend of two marginals does not make them right — it makes them blunter.
- Nothing here re-opens the budget ladder. MEDIUM remains the default and the maximum-strength candidate.
- No architecture change, no belief training, no additional search rung, no larger lambda sweep, and no Phase 14 task was touched.

