# Phase 12 Agent 2 — Belief-to-Decision Diagnostic

Generated 2026-08-20T06:50:28Z by `scripts/run_phase12_agent02.py`.

Engineering artifact of the Phase 12 rapid search-engineering phase. Position-level comparison only: no match set, no tournament, no significance claim, no change to the Agent 1 search core.

## 1. Question and verdict

```text
BELIEF CHANGES DECISIONS, BUT NOT DEMONSTRABLY FOR THE BETTER
agent1c and original_phase11 choose differently, and agent1c is not
closer to the perfect-information choice at this budget
```

- On this fresh set the belief ordering is agent1c R_CE 0.9268 < original_phase11 0.9668 < remaining_count 1.0 — the Phase 11B ordering reproduces here.
- Search changes the move in 19.1% (count), 20.3% (original), 26.6% (agent1c) and 32.0% (oracle) of positions.
- agent1c and original_phase11 pick the same action at 78.5% of positions; the belief improvement does reach the decision.
- Oracle agreement: direct C1 68.0%, count 70.3%, original 72.3%, agent1c 71.9%. Every search arm beats direct C1 here, and the three are within a few positions of each other.
- agent1c minus original_phase11 oracle agreement: TINY +0.4pp (+1 of 256 positions), SMALL -0.4pp (-1 of 256 positions) — the sign flips between budgets, so at this sample size the two beliefs are not separable by this measure.
- Of the 82 positions where direct C1 and the oracle search disagree, agent1c recovers the most (31, a 37.8% fix rate, against 26 for original_phase11 and 19 for remaining_count) but also breaks the most (21 against 15 and 13), so the net is a wash: +10 against +11 and +6.

## 2. The fresh diagnostic position set

```text
artifact      phase12_diagnostic_positions_v1
positions     256 from 156 fresh games
groups        Phase9-like 64, Strategic 64, Tactical 64, Scout-rush 64
colours       red 128, blue 128
eligibility   observer to act, ply >= 12, >= 4 unresolved opponent pieces
selection     eligible decisions at the quantile midpoints of each game, 2 per game
setups        accepted library split 'validation' (neither the spent test pool nor Agent 1C's training pool)
opponents     the four accepted Phase 11 strata, unmodified
seeds         Phase 12 personalization, master 2026082002
test bank     False (never opened)
manifest      reports/phase12/agent_02_position_manifest.json  sha-of-content 9b74a1cfeab57ec7...
```

Every position replays bit-for-bit from the manifest: the rebuilt observation is required to match the digest the observer recorded while the game was played, and all 256 did.

### Position mix

```text
median ply                   72
ply range                    13-1134
median unresolved            32
unresolved range             6-40
median moved hidden          5
median legal actions         22
positions early              65
positions middle             114
positions late               77
```

## 3. Belief quality on these same positions

Measured with the accepted `R_CE` arithmetic on this fresh set, so the three providers are comparable to each other here. These are **not** Phase 11B leaderboard numbers: different positions, different games.

| provider | pieces | CE | R_CE | top-1 | sampled-world rank accuracy |
|---|---|---|---|---|---|
| remaining_count | 7,533 | 2.1614 | 1.0000 | 21.2% | 14.2% |
| original_phase11 | 7,533 | 2.0896 | 0.9668 | 24.7% | 16.9% |
| agent1c | 7,533 | 2.0032 | 0.9268 | 28.2% | 18.0% |
| oracle | — | 0 | 0 | 100% | 100.0% (true state) |

`sampled-world rank accuracy` is the fraction of hidden pieces whose rank the provider's sampled worlds actually got right — the quantity search consumes, as opposed to the marginals it is scored on.

R_CE by behaviour group:

| provider | Phase9-like | Strategic | Tactical | Scout-rush |
|---|---|---|---|---|
| remaining_count | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| original_phase11 | 0.9335 | 0.9759 | 0.9620 | 0.9964 |
| agent1c | 0.9028 | 0.9396 | 0.9298 | 0.9377 |

## 4. Decisions at SMALL (primary comparison)

| arm | move disagreement vs direct | oracle agreement | oracle agreement when it deviated | mean S(sel) − S(direct) | median | mean Q(direct) | s/move |
|---|---|---|---|---|---|---|---|
| direct C1 (no search) | 0.0% | 68.0% | — | — | — | — | — |
| search + remaining_count | 19.1% | 70.3% | 38.8% | 0.0130 | 0.0000 | -0.1029 | 0.322 |
| search + original_phase11 | 20.3% | 72.3% | 50.0% | 0.0112 | 0.0000 | -0.1304 | 0.328 |
| search + agent1c | 26.6% | 71.9% | 45.6% | 0.0225 | 0.0000 | -0.0657 | 0.327 |
| search + oracle | 32.0% | 100.0% | 100.0% | 0.1114 | 0.0000 | -0.0840 | 0.041 |

`S(sel) − S(direct)` is non-negative by construction — the direct action is always a candidate, so a search that changes the move only does so when it scores the new move higher. Its size is how much better the arm *thinks* its choice is; `Q(direct)` is each arm's own valuation of the same unchanged action, which is where world realism shows up.

### Pairwise action agreement

| | remaining_count | original_phase11 | agent1c | oracle |
|---|---|---|---|---|
| **remaining_count** | 100.0% | 82.0% | 79.7% | 70.3% |
| **original_phase11** | 82.0% | 100.0% | 78.5% | 72.3% |
| **agent1c** | 79.7% | 78.5% | 100.0% | 71.9% |
| **oracle** | 70.3% | 72.3% | 71.9% | 100.0% |

```text
all four arms chose the identical action        60.2%
the three production arms chose identically     72.3%
at least one arm left the direct C1 move        43.8%
```

### Move disagreement vs direct C1, by behaviour group

| arm | Phase9-like | Strategic | Tactical | Scout-rush |
|---|---|---|---|---|
| remaining_count | 23.4% | 12.5% | 18.8% | 21.9% |
| original_phase11 | 25.0% | 12.5% | 17.2% | 26.6% |
| agent1c | 40.6% | 14.1% | 20.3% | 31.2% |
| oracle | 40.6% | 23.4% | 31.2% | 32.8% |

### Oracle agreement, by behaviour group

| arm | Phase9-like | Strategic | Tactical | Scout-rush |
|---|---|---|---|---|
| remaining_count | 71.9% | 78.1% | 64.1% | 67.2% |
| original_phase11 | 70.3% | 79.7% | 70.3% | 68.8% |
| agent1c | 79.7% | 73.4% | 70.3% | 64.1% |
| oracle | 100.0% | 100.0% | 100.0% | 100.0% |

Direct C1's own oracle agreement by group: Phase9-like 59.4%, Strategic 76.6%, Tactical 68.8%, Scout-rush 67.2%.

### By game phase (move disagreement / oracle agreement)

| arm | early | middle | late |
|---|---|---|---|
| remaining_count | 16.9% / 75.4% | 23.7% / 65.8% | 14.3% / 72.7% |
| original_phase11 | 13.8% / 72.3% | 24.6% / 71.1% | 19.5% / 74.0% |
| agent1c | 21.5% / 76.9% | 31.6% / 70.2% | 23.4% / 70.1% |
| oracle | 26.2% / 100.0% | 35.1% / 100.0% | 32.5% / 100.0% |

### Where the arms move the decision, relative to the oracle choice

Net oracle agreement hides two opposite effects. These are the positions where direct C1 and the perfect-information search already disagree (the headroom), and the positions where they already agree (what an arm can break).

| arm | headroom positions | fixed | fix rate | broken | break rate | net |
|---|---|---|---|---|---|---|
| remaining_count | 82 | 19 | 23.2% | 13 | 7.5% | +6 |
| original_phase11 | 82 | 26 | 31.7% | 15 | 8.6% | +11 |
| agent1c | 82 | 31 | 37.8% | 21 | 12.1% | +10 |
| oracle | 82 | 82 | 100.0% | 0 | 0.0% | +82 |

### Only the positions where the arm left the direct move

| arm | positions changed | mean ΔS | mean ΔQ | prior of chosen | prior of direct |
|---|---|---|---|---|---|
| remaining_count | 49 | 0.0677 | 0.1252 | 0.153 | 0.253 |
| original_phase11 | 52 | 0.0553 | 0.1081 | 0.155 | 0.249 |
| agent1c | 68 | 0.0849 | 0.1550 | 0.145 | 0.277 |
| oracle | 82 | 0.3478 | 0.4777 | 0.115 | 0.353 |

ΔQ is the world-averaged value the arm gains by switching; ΔS adds the policy-regularization term, which is negative for every switch away from the policy's own top move — a switch has to buy more value than it gives up in prior.

## 5. Budget sensitivity

The oracle *choice* is itself a search product, so it moves with the budget: direct C1's agreement with it is 71.1% at TINY, 68.0% at SMALL. Compare arms within a budget, never numbers across budgets.

### TINY

| arm | move disagreement vs direct | oracle agreement | oracle agreement when it deviated | mean S(sel) − S(direct) | median | mean Q(direct) | s/move |
|---|---|---|---|---|---|---|---|
| direct C1 (no search) | 0.0% | 71.1% | — | — | — | — | — |
| search + remaining_count | 20.3% | 72.7% | 40.4% | 0.0156 | 0.0000 | -0.0866 | 0.122 |
| search + original_phase11 | 20.3% | 74.2% | 48.1% | 0.0137 | 0.0000 | -0.1180 | 0.124 |
| search + agent1c | 25.4% | 74.6% | 49.2% | 0.0230 | 0.0000 | -0.0568 | 0.125 |
| search + oracle | 28.9% | 100.0% | 100.0% | 0.0993 | 0.0000 | -0.0654 | 0.030 |

Pairwise action agreement:

| | remaining_count | original_phase11 | agent1c | oracle |
|---|---|---|---|---|
| **remaining_count** | 100.0% | 84.0% | 77.3% | 72.7% |
| **original_phase11** | 84.0% | 100.0% | 77.3% | 74.2% |
| **agent1c** | 77.3% | 77.3% | 100.0% | 74.6% |
| **oracle** | 72.7% | 74.2% | 74.6% | 100.0% |

## 6. Cost

| preset | worlds | depth | arms | decisions | s/move (mean) | total s |
|---|---|---|---|---|---|---|
| TINY | 8 | 4 | 4 | 1024 | 0.100 | 102 |
| SMALL | 16 | 6 | 4 | 1024 | 0.254 | 260 |

Device `cpu`, 10 torch threads. Latency is end-to-end per decision through the whole search stack.

Repeat probe: re-deciding 16 positions (64 decisions) under the same seeds reproduced every action and root score.

## 7. Interpretation

The oracle arm is the ceiling of this mechanism, not of Stratego: it is the same search, the same rollouts and the same leaf value, run on the one true world. Its distance from direct C1 is how much this search design can move a decision at all: it leaves the direct C1 move in 32.0% of positions, and those are the entire budget of decisions any belief could hope to change for the better.

Because the oracle does move decisions, the mechanism has room in it, and the interesting question is how much of that room the learned beliefs recover. Between the count baseline (70.3% oracle agreement) and the oracle itself, agent1c reaches 71.9% and the original head 72.3%, against direct C1's 68.0%. Of the 82 positions where direct C1 and the oracle search disagree, agent1c recovers 31 and breaks 21 that direct C1 already had right (net +10); original_phase11 is 26/15 (net +11) and remaining_count 19/13 (net +6).

Nothing here is a strength claim. Agreeing with the perfect-information search is agreement with a shallow greedy-rollout evaluation that happens to know the hidden ranks; it is a diagnostic, not a proof of optimality, and Agent 3's match test is what turns any of this into wins or does not.

## 8. Limitations

- Position-level diagnostic only: agreement with a perfect-information search is not a win rate and not a strength claim.
- The oracle arm collapses to a single world, so it is both the best-informed and the cheapest arm; its latency is not comparable to the belief arms'.
- 2 positions per game over 130 contributing games: positions from one game share a setup and an opening, so they are not independent samples.
- The diagnostic setups are drawn from the accepted library's 'validation' split, which is also the pool Phase 11B's dev split drew from — different seeds, different games, but Agent 1C's candidate selection saw that pool. A mild optimistic residual for agent1c, accepted for an engineering diagnostic.
- R_CE here is computed on this fresh set and is not the Phase 11B leaderboard number for the same checkpoint.
- One fixed beta (0.1) and one candidate rule, per the common contract; no tuning was attempted.

## 9. Deliverables and status

```text
stratego/search/phase12/positions.py           (new; Agent 1's modules untouched)
tests/search/test_phase12_positions.py
reports/phase12/agent_02_position_manifest.json
reports/phase12/agent_02_decisions.csv
reports/phase12/agent_02_report.md
reports/phase12/agent_02_summary.json

phase11_final_classification     FAIL
phase11b_selection               Agent1C
scientific_validation_status     not performed
oracle_available_in_production   False
phase11_test_bank_used           False
search_core_modified             False
match_set_run                    False
agent_3_launched                 False
```

Stop condition reached: the position-level comparison is complete. No match set was run and Agent 3 is not launched.
