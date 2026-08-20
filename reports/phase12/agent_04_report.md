# Phase 12 Agent 4 — Search Budget Scaling with Agent 1C

Generated 2026-08-20T16:57:17Z by `scripts/run_phase12_agent04.py`.

Engineering artifact of the Phase 12 rapid search-engineering phase. One provider, three budgets, one fixed match pack: no grid, no tuning, no sealed test bank, and no significance claim.

## 1. Question and verdict

```text
THE THREE BUDGETS DID NOT SEPARATE
practical operating point TINY (8 worlds, depth 4, 0.126 s/move median, EWR 0.6406 against direct 0.5234)
```

- Direct accepted Phase 9 C1 scored EWR 0.5234 (33 / 1 / 30) over 64 games; search + agent1c @ TINY 0.6406 (41 / 0 / 23), search + agent1c @ SMALL 0.6250 (40 / 0 / 24), search + agent1c @ MEDIUM 0.6875 (44 / 0 / 20).
- Every rung finished above direct C1, by +0.1016 to +0.1641 EWR, which reproduces Agent 3's direction on twice the boards.
- Step by step up the ladder: TINY → SMALL -0.0156 EWR for 18.7 more search seconds per game, SMALL → MEDIUM +0.0625 EWR for 37.6 more search seconds per game. The whole ladder spans 0.0625 EWR (4.0 games of 64), against a 0.10 engineering margin.
- Latency across the ladder: TINY 0.126 s median / 0.138 s p95, SMALL 0.340 s median / 0.382 s p95, MEDIUM 0.846 s median / 0.916 s p95, against 0.0017 s for direct C1.
- The strongest rung on this pack was MEDIUM at EWR 0.6875; the selected operating point is TINY, 0.0469 EWR behind it and 10.2 s of search per game.
- Stopping rule: strength_clearly_stopped_improving, useful_operating_point_already_obvious, next_preset_consumes_disproportionate_compute.

## 2. Ladder configuration

```text
artifact                phase12_agent04_budget_ladder_v1
budget_version          phase12_budget_ladder_v1
search_version          phase12_root_world_search_v1
score_definition        S(a) = Q(a) + beta * log(pi(a) + epsilon)
belief_provider         agent1c
rungs                   TINY(worlds 8, depth 4, candidates <= 8)  SMALL(worlds 16, depth 6, candidates <= 8)  MEDIUM(worlds 32, depth 8, candidates <= 8)
beta                    0.1
reference_arm           direct accepted Phase 9 C1
opponents               Phase 9 direct, Strategic, Tactical, Scout-rush
boards                  64
games_played            256
setups                  accepted library split 'validation'
rules                   EVALUATION_RULES (accepted)
master seed             2026082003
match_version           phase12_match_test_v1
test bank               False (never opened)
device                  cpu, 10 torch threads
```

Every rung plays the identical boards under identical opponent seeds and identical per-ply world seeds, with one shared Agent 1C provider instance, so two rungs differ only in the budget they spend on the same worlds.

## 3. The ladder

Effective win rate is the accepted definition — the mean per-game score, win 1, draw 0.5, loss 0.

| rung | budget | W / D / L | EWR | vs direct | delta from previous | median s/move | p95 s/move | C1 fwd/move | worlds/move |
|---|---|---|---|---|---|---|---|---|---|
| direct accepted Phase 9 C1 | — | 33 / 1 / 30 | 0.5234 | — | — | 0.002 | 0.002 | 1.0 | — |
| search + agent1c @ TINY | 8w / d4 | 41 / 0 / 23 | 0.6406 | +0.1172 | — | 0.126 | 0.138 | 311.0 | 7.9 |
| search + agent1c @ SMALL | 16w / d6 | 40 / 0 / 24 | 0.6250 | +0.1016 | -0.0156 | 0.340 | 0.382 | 861.5 | 15.7 |
| search + agent1c @ MEDIUM | 32w / d8 | 44 / 0 / 20 | 0.6875 | +0.1641 | +0.0625 | 0.846 | 0.916 | 2221.8 | 31.6 |

At 64 games per rung an EWR standard error of about 0.061 is unavoidable, and the paired per-board comparison carries about 0.055. The engineering margin below which this agent refuses to read an ordering is 0.10.

### Throughput and cost

| rung | search s/game | game wall-clock s | games/hour | player decisions/game | fwd positions/s | move-change vs direct |
|---|---|---|---|---|---|---|
| direct accepted Phase 9 C1 | 0.18 | 0.3 | 12923.8 | 101.6 | 568 | — |
| search + agent1c @ TINY | 10.25 | 10.4 | 347.5 | 81.7 | 2479 | 15.1% |
| search + agent1c @ SMALL | 28.97 | 29.2 | 123.5 | 87.5 | 2601 | 15.0% |
| search + agent1c @ MEDIUM | 66.55 | 66.8 | 53.9 | 80.6 | 2691 | 12.4% |

### Strength bought per unit of search time

| rung | EWR vs direct | extra s/game vs direct | EWR per search second | delta EWR from previous | extra s/game vs previous | EWR per extra search second |
|---|---|---|---|---|---|---|
| TINY | +0.1172 | 10.1 | +0.01164 | — | — | — |
| SMALL | +0.1016 | 28.8 | +0.00353 | -0.0156 | 18.7 | -0.00083 |
| MEDIUM | +0.1641 | 66.4 | +0.00247 | +0.0625 | 37.6 | +0.00166 |

The two efficiency columns answer different questions: the first is what searching at all buys over playing directly, the second is what climbing one rung buys over the rung below it. The stopping rule reads the second.

### EWR by opponent

| rung | Phase 9 direct | Strategic | Tactical | Scout-rush | overall |
|---|---|---|---|---|---|
| direct accepted Phase 9 C1 | 0.375 | 0.438 | 0.531 | 0.750 | 0.523 |
| search + agent1c @ TINY | 0.500 | 0.750 | 0.500 | 0.812 | 0.641 |
| search + agent1c @ SMALL | 0.500 | 0.750 | 0.500 | 0.750 | 0.625 |
| search + agent1c @ MEDIUM | 0.875 | 0.625 | 0.438 | 0.812 | 0.688 |

### EWR by colour and setup source

| rung | red | blue | p10d opponent | neutral opponent |
|---|---|---|---|---|
| direct accepted Phase 9 C1 | 0.656 | 0.391 | 0.547 | 0.500 |
| search + agent1c @ TINY | 0.750 | 0.531 | 0.656 | 0.625 |
| search + agent1c @ SMALL | 0.750 | 0.500 | 0.562 | 0.688 |
| search + agent1c @ MEDIUM | 0.906 | 0.469 | 0.656 | 0.719 |

## 4. Paired reading of the same boards

Every rung played every board, so the informative comparison is per board. A board both rungs resolved the same way carries no information about the difference between them.

| comparison | boards | better / same / worse | paired delta | standard error | size of the delta |
|---|---|---|---|---|---|
| search + agent1c @ TINY vs direct C1 | 64 | 8 / 55 / 1 | +0.1172 | 0.043 | 7.5 games of 64 |
| search + agent1c @ SMALL vs direct C1 | 64 | 10 / 50 / 4 | +0.1016 | 0.056 | 6.5 games of 64 |
| search + agent1c @ MEDIUM vs direct C1 | 64 | 15 / 44 / 5 | +0.1641 | 0.066 | 10.5 games of 64 |
| search + agent1c @ SMALL vs search + agent1c @ TINY | 64 | 5 / 53 / 6 | -0.0156 | 0.052 | 1.0 game of 64 |
| search + agent1c @ MEDIUM vs search + agent1c @ SMALL | 64 | 8 / 52 / 4 | +0.0625 | 0.054 | 4.0 games of 64 |
| search + agent1c @ MEDIUM vs search + agent1c @ TINY | 64 | 9 / 49 / 6 | +0.0469 | 0.061 | 3.0 games of 64 |

### Boards the seat never got to play

```text
boards in the pack                          64
decided before the player's first decision  3
decided within three player decisions       12
contested boards                            61
```

These boards are kept in every headline number — they are real games and every rung is charged for them equally — but they are the same result for every rung by construction, so they set a floor on how much of the pack can separate two budgets. The cause is the setup library, not the search: 47 of 64 boards place a flag on a front row, where an opening scout down an open file can reach it.

| rung | W / D / L (all) | EWR (all) | W / D / L (contested) | EWR (contested) |
|---|---|---|---|---|
| direct accepted Phase 9 C1 | 33 / 1 / 30 | 0.5234 | 33 / 1 / 27 | 0.5492 |
| search + agent1c @ TINY | 41 / 0 / 23 | 0.6406 | 41 / 0 / 20 | 0.6721 |
| search + agent1c @ SMALL | 40 / 0 / 24 | 0.6250 | 40 / 0 / 21 | 0.6557 |
| search + agent1c @ MEDIUM | 44 / 0 / 20 | 0.6875 | 44 / 0 / 17 | 0.7213 |

## 5. Stopping rule

```text
meaningful_ewr_gain                     0.1
comfortable_move_seconds                1.0
impractical_move_seconds                5.0
latency_to_strength_ratio               3.0
disproportionate_compute_multiple       1.0
```

| condition | fired | evidence |
|---|---|---|
| strength clearly stopped improving | FIRED | MEDIUM - SMALL = +0.0625 EWR against a 0.10 margin |
| latency rises much faster than strength | no | cost +1.30x for +0.62 margins of strength; the rule allows 3.0x per margin |
| human play latency impractical | no | MEDIUM decides in 0.846 s median (p95 0.916 s) against a 1.0 s comfort line and a 5.0 s practicality line |
| useful operating point already obvious | FIRED | the practical point is TINY, below the top rung played — a larger rung would be bought for a difference this match set cannot resolve |
| larger search creates instability | no | no rung reported a defect |
| next preset consumes disproportionate compute | FIRED | LARGE projects 162.7 s of search per game, against 105.8 s for the whole ladder played so far |

```text
stop scaling after MEDIUM: strength_clearly_stopped_improving, useful_operating_point_already_obvious, next_preset_consumes_disproportionate_compute
```

## 6. Selected practical operating point

```text
worlds                      8
root candidates             <= 8
depth                       4
policy regularization       S(a) = Q(a) + 0.1 * log(pi(a) + 1e-06)
belief provider             agent1c
expected move latency       0.126 s median, 0.138 s p95, 0.193 s max
quick strength result       EWR 0.6406 (41 / 0 / 23) over 64 games against direct C1 0.5234 (33 / 1 / 30)
search seconds per game     10.2
games per hour              347.5
```

cheapest rung whose EWR is within the engineering margin of the strongest rung; unstable rungs are excluded.

## 7. Cost profile

Section 6 asks for a profile before any redesign. Measured in a separate single-process pass over 12 positions from Agent 2's manifest, replayed and observation-digest-verified by the accepted apparatus, so every rung is profiled on the same positions.

| rung | s/move | forward | observation | other | C1 fwd/move | unique worlds | fwd positions/s |
|---|---|---|---|---|---|---|---|
| TINY | 0.129 | 0.68 | 0.09 | 0.22 | 308 | 8.0 | 2385 |
| SMALL | 0.333 | 0.72 | 0.10 | 0.18 | 858 | 16.0 | 2573 |
| MEDIUM | 0.832 | 0.73 | 0.10 | 0.17 | 2190 | 32.0 | 2633 |

Main observed bottleneck: c1 forward passes. The optimizations section 6 names are already in the Agent 1 engine — worlds are sampled once at the root and de-duplicated, the root C1 outputs are computed once and reused by every candidate, and rollout forwards are batched across all live (candidate, world) simulations at each ply — so there is no cheap structural win left to take, and none is needed to make the selected operating point viable.

## 8. Cross-agent reproduction check

```text
Agent 3 arm            search_agent1c
Agent 4 rung           search_agent1c_small
shared boards          32
identical boards       32
mismatching boards     0
fields compared        outcome, plies, player_decisions, c1_forwards, move_changes
```

The SMALL rung replayed Agent 3's `search_agent1c` games exactly — same outcome, same ply count, same decision count, same forward count, same move changes — on all 32 boards the two packs share. The two agents are running the same system.

## 9. Match-time boundary probe

Each seat was re-asked a sample of its own decisions on a state whose hidden opponent identities had been permuted by the accepted `permute_hidden_identities`, and required to answer identically; the search seats were additionally required to agree with the accepted direct player on what the direct Phase 9 action was.

| rung | permutation checks | assignments actually changed | answer changed | direct-agreement checks | failures |
|---|---|---|---|---|---|
| direct accepted Phase 9 C1 | 16 | 16 | 0 | 0 | 0 |
| search + agent1c @ TINY | 16 | 16 | 0 | 16 | 0 |
| search + agent1c @ SMALL | 16 | 16 | 0 | 16 | 0 |
| search + agent1c @ MEDIUM | 16 | 16 | 0 | 16 | 0 |

No production rung may change its answer under permutation; Agent 3's oracle arm was the positive control for that check and is not replayed here.

## 10. Interpretation

The ladder cost what it was expected to cost. TINY decides in 0.126 s and MEDIUM in 0.846 s — a factor of 6.7 for a factor of 7.2 in forward passes, which is the arithmetic of worlds times candidates times plies and not a surprise. Latency is the half of this agent's question that the match pack measures precisely.

Strength did not move with budget. The whole ladder spans 0.0625 EWR — 4.0 games of 64 — against an unpaired standard error near 0.061 and a 0.10 margin, so the ordering among rungs is a record of what happened and not a ranking. Buying more worlds and more depth did not buy games on this pack.

That makes the practical operating point TINY: 10.2 s of search per game, 0.126 s per decision at the median and 0.138 s at p95, EWR 0.6406 against direct C1's 0.5234. It is chosen by the stated rule — the cheapest rung not meaningfully behind the strongest — and not by picking the largest number in the EWR column.

What that choice gives up should be stated rather than buried. The strongest rung on this pack was MEDIUM at EWR 0.6875, 0.0469 EWR — 3.0 games of 64 — ahead of TINY, and it costs 0.846 s per decision against 0.126 s. The rule prefers the cheaper rung because a lead smaller than the margin has not been shown to be a lead at all; a reader who believes the pack resolves finer than that should read MEDIUM as the operating point instead, and pay the 56.3 extra search seconds per game for it.

The pack itself is the limiting instrument, and it is worth saying how. 3 of 64 boards were decided before the player seat had a single decision, and 12 within three of them: an opening scout reaching a flag placed on a front row. Those boards return the same result for every rung whatever it spends, so the effective sample separating two budgets is smaller than the headline game count.

## 11. Limitations

- 64 games per rung is an engineering sample, not a powered experiment: an EWR difference below the stated noise scale is not evidence of an ordering between budgets.
- One provider (Agent 1C), one beta, one candidate rule, one search version. The ladder moves worlds and depth together, as the instruction's presets do, so it cannot say which of the two bought or failed to buy anything.
- The rungs share boards, opponent seeds and per-ply world seeds, which removes setup variance but leaves them correlated: paired numbers and unpaired numbers must not be mixed.
- 3 boards were decided before the player's first decision and return the same result for every rung, so the pack's effective resolution is below its game count.
- Latency is single-process on cpu with the stated thread count. It is the latency of this machine and this device, not a portable number, and a parallel match harness would change it.
- The stopping-rule thresholds are engineering judgements stated in `stratego/search/phase12/budget.py`, not measurements. A reader who disagrees with the operating point should move a threshold and re-read the table.
- Setups come from the accepted library's 'validation' split, the same pool Phase 11B's dev split drew from, so a mild optimistic residual for agent1c is accepted for an engineering match test.

## 12. Deliverables and status

```text
stratego/search/phase12/budget.py           (new; Agent 1 and 3 modules untouched)
tests/search/test_phase12_budget.py
reports/phase12/agent_04_ladder_config.json
reports/phase12/agent_04_games.jsonl
reports/phase12/agent_04_games.csv
reports/phase12/agent_04_profile.json
reports/phase12/agent_04_report.md
reports/phase12/agent_04_summary.json

phase11_final_classification      FAIL
phase11b_selection                Agent1C
scientific_validation_status      not performed
oracle_available_in_production    False
phase11_test_bank_used            False
search_core_modified              False
belief_provider                   agent1c
presets_played                    TINY, SMALL, MEDIUM
budget_above_medium_used          False
selected_operating_point          TINY
production_integration_started    False
agent_5_launched                  False
```

Stop condition reached: a practical operating point has been identified. No preset above MEDIUM was run. Production integration was not begun and Agent 5 is not launched.
