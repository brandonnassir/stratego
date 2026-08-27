# Phase 15 — Agent 2 follow-up
## Deeper-search pilot: P24 + B24 at 2x and 4x MEDIUM

**Question.** does buying roughly 2-4x more search compute make the selected P24 + B24 meaningfully stronger than MEDIUM, and what does it cost?

**Answer: MEDIUM.** both stronger rungs came out *worse* than MEDIUM on the paired pack (LARGE -0.0750, XLARGE -0.0750), so the extra compute does not merely fail to pay for itself — the point estimate is a regression, and MEDIUM stands.

> The honest one-line version: buying 2.19x and 3.90x more search compute did not make P24 + B24 stronger. It made it **weaker** — and the oracle control in section 3 shows why, which is the part of this result worth carrying forward.

This is a narrow paired pilot on one already-selected system. No architecture was changed, no network was trained, no belief experiment was broadened, and no Phase 14 task was touched. The ladder is closed here.

## 1. What was varied, and what was held fixed

Compute grows through **worlds first, depth modestly** — everything else is MEDIUM's. `LARGE` and `XLARGE` do not pass `max_root_candidates`, `beta` or `epsilon` at all, so they inherit the same defaults MEDIUM uses; the pilot's own control is therefore a property of the configuration rather than a promise, and the gate checks it.

| rung | worlds | depth | candidates | beta | planned compute | measured compute | forwards/move | world uniqueness |
|---|---|---|---|---|---|---|---|---|
| MEDIUM | 32 | 8 | 8 | 0.1 | 1.00x | 1.00x | 1964 | 0.926 |
| LARGE | 64 | 9 | 8 | 0.1 | 2.25x | 2.19x | 4302 | 0.909 |
| XLARGE | 96 | 11 | 8 | 0.1 | 4.12x | 3.90x | 7651 | 0.898 |

Measured compute lands at **2.19x** and **3.90x** — slightly under the planned 2.25x and 4.12x, because duplicate sampled worlds are evaluated once and weighted, and duplicates get commoner as the world budget grows (uniqueness falls from 0.926 to 0.898). The measured ratio is the honest one and is what the cost figures below use.

These rungs are deliberately **not** the section 7 `STRONG` preset, which raises the candidate count to 12: this pilot forbids changing candidate handling, so reusing `STRONG` would have quietly broken its own control.

## 2. Integrity at the larger budgets

**PASS** — every check ran on fresh replayed positions before a single deeper game was played.

| check | result | observed |
|---|---|---|
| frozen identity | pass | P24 `622d9e6caa723c93` + B24 `ac5e15b87f5c5cfd`, temperature 1.0 |
| configuration control | pass | candidates, beta, epsilon and world dedup identical to MEDIUM at every rung |
| determinism and legality | pass | 24 decisions re-run under the same seed at every rung: identical action, identical world weights, identical Q values; every action legal; the direct move always a candidate |
| sampled worlds legal | pass | LARGE 512, MEDIUM 256, XLARGE 768 worlds through the accepted validation stack |
| MEDIUM reproduces Stage C | pass | 60 boards played twice in separate runs, hours apart: identical outcome, score and ply count |

## 3. The paired pack

360 games on the same **60 balanced boards** — one per cell of the 10 opponents x 3 setup sources x 2 colours grid — with the same opponents, the same setups and the same per-decision seeds at every rung. All three rungs were replayed fresh in this one pack, so every paired delta comes from rows produced by identical code under identical conditions. 9.7 hours on 10 workers.

| rung | W/D/L | EWR | paired vs MEDIUM | worst opponent | mean move (idle) | median move (idle) | p95 move (idle) | % moves differing | fallbacks/errors |
|---|---|---|---|---|---|---|---|---|---|
| MEDIUM | 55/2/3 | 0.9333 | — (baseline) | 0.833 (p18) | 1.484 | 1.713 | 1.775 | 0.000 | 0 |
| LARGE | 51/1/8 | 0.8583 | -0.0750 ± 0.0529 | 0.667 (p24) | 3.250 | 3.818 | 3.918 | 0.150 | 0 |
| XLARGE | 51/1/8 | 0.8583 | -0.0750 ± 0.0409 | 0.500 (p24) | 5.738 | 6.792 | 6.989 | 0.125 | 0 |

`% moves differing` is measured on the fixed diagnostic position manifest, where every rung answers the same question. Inside a match game the quantity is ill-defined: the moment a rung plays a different move the two games diverge and later positions are no longer comparable, so a per-ply count there would measure divergence of *positions*, not of decisions.

### How far into a game the extra search takes to matter

| rung | games compared | identical to MEDIUM | diverged | median first-divergence ply | earliest |
|---|---|---|---|---|---|
| LARGE | 60 | 5 | 55 | 7 | 0 |
| XLARGE | 60 | 1 | 59 | 8 | 0 |

### Latency, idle and in-pack

| rung | median (idle) | p95 (idle) | max (idle) | median (10-way) | p95 (10-way) | search s/game (10-way) |
|---|---|---|---|---|---|---|
| MEDIUM | 1.713 | 1.775 | 1.795 | 3.155 | 3.267 | 722.9 |
| LARGE | 3.818 | 3.918 | 3.961 | 6.889 | 7.142 | 1618.9 |
| XLARGE | 6.792 | 6.989 | 7.076 | 12.174 | 12.578 | 3055.2 |

The idle column is what a person playing one game experiences and is the column the 5 s ceiling is applied to.

### EWR by opponent

| rung | p18 | p24 | phase9_anchor | strategic_rule_based | tactical_rule_based | stress_scout_rush | stress_miner_rush | stress_berserker | stress_information_miser | stress_chaos |
|---|---|---|---|---|---|---|---|---|---|---|
| MEDIUM | 0.833 | 0.833 | 0.833 | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| LARGE | 0.750 | 0.667 | 0.833 | 0.833 | 1.000 | 1.000 | 0.667 | 0.833 | 1.000 | 1.000 |
| XLARGE | 0.750 | 0.500 | 0.667 | 1.000 | 1.000 | 0.833 | 1.000 | 0.833 | 1.000 | 1.000 |

### Oracle ceiling at each budget (offline diagnostic)

Essentially free to run: the oracle's sampled worlds all collapse to the one true army, so its cost does not grow with the world budget. It is never a deployable arm and is excluded from production by four independent refusals.

| rung | oracle EWR | paired vs oracle MEDIUM | worst opponent | search s/game |
|---|---|---|---|---|
| MEDIUM | 0.8833 | — (baseline) | 0.667 | 18.2 |
| LARGE | 0.9250 | 0.0417 ± 0.0342 | 0.833 | 26.1 |
| XLARGE | 0.9083 | 0.0250 ± 0.0384 | 0.750 | 29.5 |

**This is the pilot's most informative result, and it inverts the main one.** Given the *true* hidden army, more search **helps**: the oracle gains +0.042 at LARGE and +0.025 at XLARGE, and its worst opponent improves from 0.667 to 0.833. Given *belief-sampled* worlds, the same extra compute **hurts**, by −0.075 at both rungs, with the worst opponent falling from 0.833 to 0.500.

So the search mechanics, the rollout policy and the leaf value are not what is failing at depth — over correct worlds they scale the way one would hope. What does not survive scaling is the *world distribution*. Averaging over 64-96 sampled worlds instead of 32, and rolling each out 9-11 plies instead of 8, commits harder and for longer to a belief that is wrong in a correlated way, so the extra compute buys a more confident wrong answer rather than a better one. That reading is consistent with the rest of the phase: the belief specialists never separated from the count baseline at MEDIUM either.

## 4. The decision

Rule: recommend the stronger rung when it gains >= the meaningful band and its p95 move fits the ceiling; keep MEDIUM when the gain is tiny or noisy while latency multiplies; prefer LARGE when LARGE improves and XLARGE does not.

| quantity | LARGE | XLARGE |
|---|---|---|
| paired gain vs MEDIUM | -0.0750 | -0.0750 |
| standard error | 0.0529 | 0.0409 |
| p95 move, idle | 3.918 | 6.989 |
| fits the 5.0 s ceiling | yes | no |

Meaningful-gain band: 0.03 to 0.05 EWR.

**Recommendation: MEDIUM.** both stronger rungs came out *worse* than MEDIUM on the paired pack (LARGE -0.0750, XLARGE -0.0750), so the extra compute does not merely fail to pay for itself — the point estimate is a regression, and MEDIUM stands.

### How far this sample can be trusted

The regressions are 1.4 and 1.8 standard errors from zero. **Neither is individually resolved at 60 paired boards**, and this pilot makes no significance claim. What raises it above noise is that four things point the same way at once: both rungs regress by the same amount, the worst-opponent score falls monotonically with compute (0.833 to 0.667 to 0.500), games get *longer* rather than more decisive, and the oracle control moves in the opposite direction on the same boards with the same seeds. A single noisy arm would not produce that pattern.

The conservative reading is the one the decision rule already takes: there is no evidence of a gain, so do not spend the compute. The stronger reading — that belief-sampled search actively degrades with scale — is supported but not established here, and would need a larger pack to settle. That pack is not part of this pilot.

## 5. What this pilot did not do

- it did not change the algorithm, the candidate rule or the regularization;
- it did not train or modify any network;
- it did not broaden the belief experiments;
- it did not touch any Phase 14 task or artifact;
- it did not extend the ladder beyond these three rungs, and will not.

