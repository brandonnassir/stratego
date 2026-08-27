# Phase 15 — Agent 2
## P18/P24 belief-guided search integration

**P24 + B24** at **TINY** (8 worlds, <= 8 candidates, depth 4), maximum-strength mode **MEDIUM**.

This is an engineering deliverable, not a scientific claim. `scientific_validation_status: not performed`. No significance claim is made anywhere in this report; every table carries its own game count.

## 1. What was built

The accepted Phase 12 engine (`phase12_root_world_search_v1`) run over the two frozen Phase 14 move models and the two Phase 15 belief specialists. The algorithm is unchanged: root-sampled worlds, a fixed candidate set, greedy rollouts for both sides, and the score `S(a) = Q(a) + beta * log(pi(a) + epsilon)`. What Phase 15 changes is which models the algorithm runs on.

| role | model |
|---|---|
| root policy / candidate prior | P24 |
| rollout policy, both sides | P24 |
| leaf value | P24 |
| direct fallback | P24 |
| hidden-rank marginals | B24 |
| legal hidden-world sampling | B24 |

The selected system is **not** a cross-pairing: `p24_b24` computes its marginals over **P24**'s frozen prefix and runs its policy and value on **P24**, the same model. The two cross-pairings were built and measured — section 4 requires it — and the matrix in section 7 reports them; they simply did not win.

## 2. Fresh, orientation-safe evidence

No Phase 12 board is reused. All 120 match boards and all 120 diagnostic positions were drawn afresh and left through the accepted orientation helper, then passed Agent 1's whole section 4 board gate — flag row, legal setup rows, exact inventory, paired Red/Blue mirror — before describing a game.

- orientation rule: `red engine row == canonical rank; blue engine row == 9 - canonical rank`
- match manifest digest: `f2e2e7a4504ea2712ddef7d7d8429e147a7a6c57eb2aeb5b19adb5e368a5e76b`
- position manifest digest: `01d1ba603287cddacb0fb61c9c24504d80e0a55e3eedbc3d3c9355293da06c49`
- setup library split: `validation`
- balance: neutral_v1 40, phase14_learned 40, targeted_family 40; colours blue 60, red 60

## 3. The correctness gate (section 9)

**PASS** — 10/10 checks, 51.151s.

| check | result | observed |
|---|---|---|
| decisions | pass | 48 decisions, 376 candidates, 356 worlds |
| fallback | pass | 48 timeout and 48 forced-error fallbacks each returned the correct direct move |
| gate_positions | pass |  |
| identities | pass |  |
| latency_probe | pass |  |
| model_roles | pass | direct action provider-invariant on 24 checks; search differed by provider on 24; P18/P24 disagreed on 3 of 12 positions |
| oracle_refusals | pass | 6 independent refusals |
| permutation_invariance | pass | 48 production checks unchanged; oracle control sensitive on 6/24 |
| phase12_frozen_candidate_regression | pass | 131 passed in 2.09s |
| worlds_legal | pass |  |

## 4. Stage A — the decision diagnostic (section 11)

120 replayed positions, every arm on the same position with the same seed, preset `TINY`.

| arm | move change | oracle agree | legal | median s | p95 s | forwards | world uniq | score margin |
|---|---|---|---|---|---|---|---|---|
| p18_direct | 0.000 | 0.875 | 1.000 | 0.002 | 0.002 | 1 | - | - |
| p24_direct | 0.000 | 0.900 | 1.000 | 0.002 | 0.002 | 1 | - | - |
| p18_remaining_count | 0.092 | 0.858 | 1.000 | 0.118 | 0.141 | 294 | 0.985 | 0.2016 |
| p18_b18 | 0.067 | 0.908 | 1.000 | 0.121 | 0.141 | 286 | 0.978 | 0.1966 |
| p18_b24 | 0.075 | 0.933 | 1.000 | 0.122 | 0.152 | 285 | 0.978 | 0.1870 |
| p24_remaining_count | 0.075 | 0.933 | 1.000 | 0.119 | 0.165 | 291 | 0.985 | 0.2303 |
| p24_b18 | 0.100 | 0.925 | 1.000 | 0.122 | 0.154 | 283 | 0.978 | 0.2640 |
| p24_b24 | 0.108 | 0.908 | 1.000 | 0.121 | 0.127 | 282 | 0.978 | 0.2597 |
| p18_oracle | 0.125 | 1.000 | 1.000 | 0.030 | 0.031 | 36 | 0.125 | 0.1986 |
| p24_oracle | 0.100 | 1.000 | 1.000 | 0.029 | 0.031 | 36 | 0.125 | 0.1878 |

**P18: `learned_belief_tracks_oracle`** — the learned belief changes decisions at a rate comparable to the oracle.

**P24: `learned_belief_tracks_oracle`** — the learned belief changes decisions at a rate comparable to the oracle.

## 5. Stage B — the complete-system match comparison (section 12)

1200 games: 10 arms on the same 120 paired boards, preset `TINY`, 130.6 minutes on 10 workers.

| arm | W/D/L | EWR | paired vs direct | worst opponent | weakness pack | median s/move | p95 s/move | move change | fallback |
|---|---|---|---|---|---|---|---|---|---|
| p18_b18\|TINY | 99/1/20 | 0.8292 | 0.0583 ± 0.0432 | 0.500 (p24) | 0.786 | 0.453 | 0.462 | 0.067 | 0.00000 |
| p18_b24\|TINY | 95/3/22 | 0.8042 | 0.0333 ± 0.0413 | 0.375 (p24) | 0.773 | 0.455 | 0.468 | 0.068 | 0.00000 |
| p18_direct\|direct | 90/5/25 | 0.7708 | - | 0.458 (p24) | 0.799 | 0.002 | 0.002 | - | 0.00000 |
| p18_oracle\|TINY | 100/9/11 | 0.8708 | 0.1000 ± 0.0400 | 0.667 (p24) | 0.838 | 0.050 | 0.051 | 0.098 | 0.00000 |
| p18_remaining_count\|TINY | 102/4/14 | 0.8667 | 0.0958 ± 0.0385 | 0.708 (p24) | 0.831 | 0.450 | 0.464 | 0.065 | 0.00000 |
| p24_b18\|TINY | 94/4/22 | 0.8000 | 0.0625 ± 0.0395 | 0.583 (p18) | 0.792 | 0.453 | 0.466 | 0.064 | 0.00000 |
| p24_b24\|TINY | 103/4/13 | 0.8750 | 0.1375 ± 0.0414 | 0.708 (p18) | 0.909 | 0.452 | 0.463 | 0.065 | 0.00000 |
| p24_direct\|direct | 87/3/30 | 0.7375 | - | 0.333 (p24) | 0.766 | 0.002 | 0.002 | - | 0.00000 |
| p24_oracle\|TINY | 105/2/13 | 0.8833 | 0.1458 ± 0.0452 | 0.750 (p24) | 0.864 | 0.049 | 0.051 | 0.092 | 0.00000 |
| p24_remaining_count\|TINY | 93/4/23 | 0.7917 | 0.0542 ± 0.0418 | 0.542 (p24) | 0.786 | 0.455 | 0.465 | 0.063 | 0.00000 |

### EWR by opponent

| arm | p18 | p24 | phase9_anchor | strategic_rule_based | tactical_rule_based | stress_scout_rush | stress_miner_rush | stress_berserker | stress_information_miser | stress_chaos |
|---|---|---|---|---|---|---|---|---|---|---|
| p18_b18\|TINY | 0.667 | 0.500 | 0.750 | 0.917 | 0.833 | 0.958 | 0.917 | 0.833 | 1.000 | 0.917 |
| p18_b24\|TINY | 0.583 | 0.375 | 0.583 | 0.833 | 0.833 | 0.917 | 1.000 | 1.000 | 1.000 | 0.917 |
| p18_direct\|direct | 0.667 | 0.458 | 0.667 | 0.917 | 0.750 | 0.875 | 0.667 | 0.833 | 0.958 | 0.917 |
| p18_oracle\|TINY | 0.750 | 0.667 | 0.917 | 1.000 | 0.833 | 0.833 | 0.917 | 0.917 | 0.958 | 0.917 |
| p18_remaining_count\|TINY | 0.792 | 0.708 | 1.000 | 0.917 | 0.833 | 1.000 | 0.750 | 0.750 | 1.000 | 0.917 |
| p24_b18\|TINY | 0.583 | 0.750 | 0.667 | 0.750 | 0.875 | 0.875 | 0.750 | 0.833 | 1.000 | 0.917 |
| p24_b24\|TINY | 0.708 | 0.750 | 0.917 | 0.833 | 0.750 | 0.958 | 1.000 | 0.917 | 1.000 | 0.917 |
| p24_direct\|direct | 0.500 | 0.333 | 0.500 | 0.833 | 0.833 | 0.875 | 0.667 | 0.917 | 1.000 | 0.917 |
| p24_oracle\|TINY | 0.833 | 0.750 | 0.833 | 1.000 | 0.750 | 0.917 | 0.917 | 0.917 | 1.000 | 0.917 |
| p24_remaining_count\|TINY | 0.625 | 0.542 | 0.583 | 0.750 | 0.750 | 0.917 | 0.917 | 0.917 | 1.000 | 0.917 |

### Is the learned belief useful? (sections 10 and 17)

This is the question the `remaining_count` control and the `oracle` ceiling exist to answer, and the pack answers it in three parts.

| move model | direct | count control | best learned | oracle ceiling | learned - count | share of ceiling recovered |
|---|---|---|---|---|---|---|
| P18 | 0.7708 | 0.8667 | 0.8292 (b18) | 0.8708 | -0.0375 | 0.58 |
| P24 | 0.7375 | 0.7917 | 0.8750 (b24) | 0.8833 | 0.0833 | 0.94 |

**Search helps.** Every search arm beats its own direct model on the paired boards, by +0.033 to +0.146 EWR. For P24, B24 recovers 94% of what perfect hidden-piece knowledge buys; for P18 the best *learned* arm recovers 58%, while the count control recovers 96%.

**But the ceiling is low.** The oracle - which reads the true hidden army - is worth only +0.100 (P18) and +0.146 (P24) over direct play at this budget. Most of the headroom in this design is not in belief quality, which is the same reading Stage A gave from the other side: even perfect information changed only 10-12% of decisions.

**And the learned belief does not consistently beat the count baseline.** For P18 the count control (0.8667) beats both specialists; for P24, B24 (0.8750) beats the count control (0.7917). The ordering flips with the move model, and the pairwise gaps are one to two standard errors on 120 paired boards - this pack cannot resolve them. The selected system is therefore the strongest *complete system among the four P/B combinations section 14 asks about*, not a demonstration that a learned belief head is required: `p18_remaining_count`, which uses no learned belief at all, scores within 0.01 EWR of it.

The one signal that does point at the specialists is Stage A's oracle agreement: for P18, count-guided search agrees with the oracle's move *less* often (0.858) than doing nothing at all (0.875), while B18 and B24 raise it to 0.908 and 0.933. Count-based worlds move the decision away from what perfect information would choose; the learned worlds move it toward. That is a decision-level observation on 120 positions, and it did not convert into a match-level separation here.

Match-time probe: all clear — p18_b18 115 permutation checks; p18_b24 122 permutation checks; p18_direct 159 permutation checks; p18_oracle 107 permutation checks, 10 sensitive (oracle control); p18_remaining_count 97 permutation checks; p24_b18 114 permutation checks; p24_b24 101 permutation checks; p24_direct 132 permutation checks; p24_oracle 81 permutation checks, 12 sensitive (oracle control); p24_remaining_count 93 permutation checks.

## 6. Stage C — the budget ladder (section 13)

240 games: 2 pairing(s) × 3 presets on the same 60 boards and seeds, 197.3 minutes.

### p18_b18

| preset | budget | EWR | paired vs TINY | search s/game | EWR/added s | median s/move | p95 s/move | human play |
|---|---|---|---|---|---|---|---|---|
| TINY | 8w/d4 | 0.8333 | 0.0000 | 102.43 | - | 0.454 | 0.461 | comfortable |
| SMALL | 16w/d6 | 0.8667 | 0.0333 | 247.54 | 0.00023 | 1.288 | 1.373 | comfortable |
| MEDIUM | 32w/d8 | 0.8417 | 0.0083 | 632.49 | -0.00006 | 3.059 | 3.182 | acceptable |

Selected **TINY** (strongest observed: SMALL at 0.8667). walk the ladder from the cheapest rung and stop at the first that is within the engineering margin of the strongest observed rung, does not regress the weakness pack, and is not impractical for human play.

Note that **SMALL is better on every measured axis** here — overall EWR, worst opponent and the weakness pack all rise from TINY. It was not selected as the default because the gain sits inside the 0.10 engineering margin and the rule prefers the cheaper rung, not because it was found no stronger. Choosing TINY as the default is a cost decision; SMALL is retained as the maximum-strength mode for callers who will pay for it.

STRONG gate: **refused** — MEDIUM did not show a useful improvement over the cheaper rungs (MEDIUM improvement over the cheaper rungs -0.0250, required 0.1000).

### p24_b24

| preset | budget | EWR | paired vs TINY | search s/game | EWR/added s | median s/move | p95 s/move | human play |
|---|---|---|---|---|---|---|---|---|
| TINY | 8w/d4 | 0.8667 | 0.0000 | 102.39 | - | 0.450 | 0.461 | comfortable |
| SMALL | 16w/d6 | 0.9000 | 0.0333 | 292.60 | 0.00017 | 1.254 | 1.318 | comfortable |
| MEDIUM | 32w/d8 | 0.9333 | 0.0667 | 708.07 | 0.00008 | 3.114 | 3.208 | acceptable |

Selected **TINY** (strongest observed: MEDIUM at 0.9333). walk the ladder from the cheapest rung and stop at the first that is within the engineering margin of the strongest observed rung, does not regress the weakness pack, and is not impractical for human play.

Note that **MEDIUM is better on every measured axis** here — overall EWR, worst opponent and the weakness pack all rise from TINY. It was not selected as the default because the gain sits inside the 0.10 engineering margin and the rule prefers the cheaper rung, not because it was found no stronger. Choosing TINY as the default is a cost decision; MEDIUM is retained as the maximum-strength mode for callers who will pay for it.

STRONG gate: **refused** — MEDIUM did not show a useful improvement over the cheaper rungs (MEDIUM improvement over the cheaper rungs 0.0333, required 0.1000).

## 7. The system matrix and the selection (section 14)

| system | direct EWR | search EWR | paired delta | worst stratum | weakness pack | median/p95 move | fallback |
|---|---|---|---|---|---|---|---|
| p18_b18 | 0.7708 | 0.8292 | 0.0583 ± 0.0432 | 0.500 (p24) | 0.786 | 0.453 / 0.462 | 0.00000 |
| p18_b24 | 0.7708 | 0.8042 | 0.0333 ± 0.0413 | 0.375 (p24) | 0.773 | 0.455 / 0.468 | 0.00000 |
| p24_b18 | 0.7375 | 0.8000 | 0.0625 ± 0.0395 | 0.583 (p18) | 0.792 | 0.453 / 0.466 | 0.00000 |
| p24_b24 | 0.7375 | 0.8750 | 0.1375 ± 0.0414 | 0.708 (p18) | 0.909 | 0.452 / 0.463 | 0.00000 |

Decision rule: composite = 0.5*overall EWR + 0.2*worst opponent + 0.15*worst family + 0.15*weakness-pack family EWR; ties inside the engineering margin broken by median latency bucketed to 0.05s, then the simpler pairing, then the composite.

Selected: **p24_b24** at **TINY**; maximum strength **MEDIUM**. Contenders inside the 0.10 engineering margin: p24_b24, p24_b18, p18_b18.

## 8. The working player (section 15)

| mode | system | budget | time cap |
|---|---|---|---|
| `p18_direct` | P18 | direct, no search | - |
| `p24_direct` | P24 | direct, no search | - |
| `selected_search` | P24 + B24 | TINY: 8 worlds, depth 4 | 0.91s |
| `maximum_strength` | P24 + B24 | MEDIUM: 32 worlds, depth 8 | 5.0s |

`oracle_available_in_production = false`. Fallback: on timeout, search error, non-finite score or an illegal result, play the same move model's direct legal move; never forfeit.

### Latency, measured twice

The Stage B and Stage C move times are measured with ten worker processes on fourteen cores, so every one of them carries scheduler contention. A person playing one game has the machine to themselves. Both numbers are kept, and the time caps are set from the un-contended pilot: a cap derived from the contended numbers would buy headroom the deployed player does not need, and would let a real stall pass unnoticed.

| preset | median (idle) | p95 (idle) | max (idle) | median (10-way) | p95 (10-way) | forwards/move |
|---|---|---|---|---|---|---|
| TINY | 0.248 | 0.259 | 0.298 | 0.450 | 0.461 | 283 |
| SMALL | 0.708 | 0.765 | 0.773 | 1.254 | 1.318 | 779 |
| MEDIUM | 1.709 | 1.782 | 1.802 | 3.114 | 3.208 | 1964 |

Measured on 40 replayed diagnostic positions, one process. Both selected presets sit inside the 2 s *preferred* line on an idle machine, and every measured move — idle or contended — stays inside its cap. Contention costs about 1.8x: MEDIUM's p95 goes from 1.78 s idle to 3.21 s under ten-way load, which is still under the 5.0 s cap but leaves 1.6x headroom rather than the 2.8x the idle measurement gives. A deployed player is not competing with nine copies of itself, so the idle column is the one a human experiences.

The production pairing ids remain available as diagnostic mode names for machine evaluation; `oracle` is not among them and is refused by name.

## 9. Known limitations

- a compact engineering pack: no significance claim is made and the per-arm sample is 120 paired boards
- the learned belief specialists did NOT consistently beat the remaining-count control: for P18 the count arm scored higher than both specialists, and for P24 only B24 beat it. The selected system is the strongest of the four P/B combinations, not evidence that a learned belief head is required; p18_remaining_count scored within 0.01 EWR of the selection while using no learned belief at all
- the oracle ceiling is small at this budget (+0.100 EWR for P18, +0.146 for P24), so most of the headroom in this search design is not in hidden-piece inference quality
- the match boards draw from the accepted setup library's `validation` split, which is also the population Agent 1's calibration and development corpora drew from; B18/B24 weights saw only the `train` split, so no belief model trained on these boards, but the two measurements are not independent draws of the base population
- the oracle arm is an offline ceiling diagnostic and is excluded from production by four independent refusals
- Stage A is a decision diagnostic on replayed positions, not a strength measurement
- no scientific validation phase was performed; this is an engineering selection

## 10. What this run did not do

- it did not train or modify B18, B24, P18 or P24;
- it did not alter, pause, stop, restart or finalize any Phase 14 task;
- it did not edit accepted Phase 12 behaviour or overwrite `phase12_search_candidate_v1`;
- it did not reuse any contaminated Phase 12 board as new evidence;
- it did not perform a scientific validation phase.

