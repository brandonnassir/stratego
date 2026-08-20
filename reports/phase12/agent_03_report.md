# Phase 12 Agent 3 — First Search Match Test

Generated 2026-08-20T08:23:33Z by `scripts/run_phase12_agent03.py`.

Engineering artifact of the Phase 12 rapid search-engineering phase. A compact match test: whole games, one budget, no tuning, no sealed test bank, and no significance claim.

## 1. Question and verdict

```text
SEARCH BEAT DIRECT C1 ON THIS 32-GAME SET AT THE SMALL BUDGET
every search arm was ahead (+0.1094 to +0.1406 EWR); the three belief providers did not separate from each other
```

- Direct accepted Phase 9 C1 scored EWR 0.5156 (16 / 1 / 15) over 32 games; search + remaining_count 0.6562 (21 / 0 / 11), search + original_phase11 0.6562 (21 / 0 / 11), search + agent1c 0.6250 (20 / 0 / 12).
- The offline oracle arm — the same search, the same rollouts and the same leaf value on the one true world — scored 0.7188 (23 / 0 / 9), +0.2031 against direct C1 and +0.0625 against the best production arm.
- The three belief providers spread 0.0312 EWR — 1.0 game of 32 — from search + remaining_count down to search + agent1c, so this match set does not separate them. Agent 2's position-level diagnostic reached the same conclusion by a different route.
- Paired against direct C1 on the same boards: search + remaining_count 7 better / 23 same / 2 worse, search + original_phase11 5 better / 26 same / 1 worse, search + agent1c 4 better / 27 same / 1 worse. 23-27 of 32 boards ended the same way whichever arm played them, which is why the paired standard error (0.075) is tighter than the unpaired one (0.085).
- Two slices of the same boards disagree. Head to head against the direct player itself (the Phase 9 direct stratum, 8 games): search + remaining_count 0.750, search + original_phase11 0.625, search + agent1c 0.375. Against the three rule opponents only: search + remaining_count 0.625, search + original_phase11 0.667, search + agent1c 0.708 (direct C1 0.604). The production ordering reverses between them.
- Search changed the direct move in 13.7% (search + remaining_count), 13.7% (search + original_phase11), 15.1% (search + agent1c) of its decisions, at 0.322 s/move (search + remaining_count), 0.322 s/move (search + original_phase11), 0.322 s/move (search + agent1c) against 0.0018 s/move for direct C1.

## 2. Match configuration

```text
artifact        phase12_agent03_match_set_v1
search version  phase12_root_world_search_v1
budget          SMALL  worlds 16  root moves <= 8  depth 6
score           S(a) = Q(a) + beta * log(pi(a) + epsilon)  beta 0.1
opponents       Phase 9 direct, Strategic, Tactical, Scout-rush
boards          32 (8 per opponent, 2 sources x 2 colours)
arms            5  (160 games played)
setups          accepted library split 'validation'
rules           EVALUATION_RULES (accepted)
master seed     2026082003
test bank       False (never opened)
device          cpu, 10 torch threads
```

Every arm played the identical boards under identical opponent seeds and identical per-ply search seeds; the arms differ only in the player seat. The match identity names an arm-independent player on purpose, so the opponent's frozen seed cannot vary with the arm under test.

## 3. Results

Effective win rate is the accepted definition — the mean per-game score, win 1, draw 0.5, loss 0.

| arm | W / D / L | EWR | vs direct | paired boards +/=/− | s/move | s/game |
|---|---|---|---|---|---|---|
| direct accepted Phase 9 C1 | 16 / 1 / 15 | 0.5156 | — | — | 0.002 | 0.3 |
| search + remaining_count | 21 / 0 / 11 | 0.6562 | +0.1406 | 7/23/2 | 0.322 | 28.8 |
| search + original_phase11 | 21 / 0 / 11 | 0.6562 | +0.1406 | 5/26/1 | 0.322 | 31.2 |
| search + agent1c | 20 / 0 / 12 | 0.6250 | +0.1094 | 4/27/1 | 0.322 | 27.8 |
| search + oracle (diagnostic) | 23 / 0 / 9 | 0.7188 | +0.2031 | 8/23/1 | 0.040 | 3.1 |

At 32 games per arm an EWR standard error of about 0.085 is unavoidable, and the paired per-board comparison carries about 0.075. Differences smaller than the 0.10 engineering margin are not read as an ordering here.

### EWR by opponent

| arm | Phase 9 direct | Strategic | Tactical | Scout-rush | overall |
|---|---|---|---|---|---|
| direct accepted Phase 9 C1 | 0.250 | 0.625 | 0.562 | 0.625 | 0.516 |
| search + remaining_count | 0.750 | 0.500 | 0.625 | 0.750 | 0.656 |
| search + original_phase11 | 0.625 | 0.750 | 0.500 | 0.750 | 0.656 |
| search + agent1c | 0.375 | 0.875 | 0.500 | 0.750 | 0.625 |
| search + oracle (diagnostic) | 0.750 | 0.750 | 0.625 | 0.750 | 0.719 |

W / D / L by opponent:

| arm | Phase 9 direct | Strategic | Tactical | Scout-rush |
|---|---|---|---|---|
| direct accepted Phase 9 C1 | 2 / 0 / 6 | 5 / 0 / 3 | 4 / 1 / 3 | 5 / 0 / 3 |
| search + remaining_count | 6 / 0 / 2 | 4 / 0 / 4 | 5 / 0 / 3 | 6 / 0 / 2 |
| search + original_phase11 | 5 / 0 / 3 | 6 / 0 / 2 | 4 / 0 / 4 | 6 / 0 / 2 |
| search + agent1c | 3 / 0 / 5 | 7 / 0 / 1 | 4 / 0 / 4 | 6 / 0 / 2 |
| search + oracle (diagnostic) | 6 / 0 / 2 | 6 / 0 / 2 | 5 / 0 / 3 | 6 / 0 / 2 |

### EWR by colour and setup source

| arm | red | blue | p10d opponent | neutral opponent |
|---|---|---|---|---|
| direct accepted Phase 9 C1 | 0.750 | 0.281 | 0.531 | 0.500 |
| search + remaining_count | 0.875 | 0.438 | 0.688 | 0.625 |
| search + original_phase11 | 0.938 | 0.375 | 0.625 | 0.688 |
| search + agent1c | 0.750 | 0.500 | 0.562 | 0.688 |
| search + oracle (diagnostic) | 0.875 | 0.562 | 0.688 | 0.750 |

## 4. Highlighted comparisons

**Agent1C search vs direct C1**

```text
EWR            0.6250 vs 0.5156   delta +0.1094
W / D / L      20/0/12 vs 16/1/15
paired boards  4 better / 27 same / 1 worse over 32 boards
paired delta   +0.1094 (standard error 0.062)
```

**Agent1C search vs old-belief search**

```text
EWR            0.6250 vs 0.6562   delta -0.0312
W / D / L      20/0/12 vs 21/0/11
paired boards  2 better / 27 same / 3 worse over 32 boards
paired delta   -0.0312 (standard error 0.071)
```

**Agent1C search vs remaining-count search**

```text
EWR            0.6250 vs 0.6562   delta -0.0312
W / D / L      20/0/12 vs 21/0/11
paired boards  3 better / 25 same / 4 worse over 32 boards
paired delta   -0.0312 (standard error 0.084)
```

**Two readings of the same 32 boards**

In the `Phase 9 direct` stratum the opponent *is* the arm-A player, so those eight boards are a head-to-head against direct C1 for a search arm — and a mirror of itself for arm A, where the result records the board rather than the player. The other 24 boards are the three rule opponents. The two slices do not agree, which is the clearest statement of how little this match set separates:

| arm | head-to-head vs direct C1 (8) | EWR there | vs the 3 rule opponents (24) |
|---|---|---|---|
| direct accepted Phase 9 C1 | 2 / 0 / 6 | 0.250 | 0.604 |
| search + remaining_count | 6 / 0 / 2 | 0.750 | 0.625 |
| search + original_phase11 | 5 / 0 / 3 | 0.625 | 0.667 |
| search + agent1c | 3 / 0 / 5 | 0.375 | 0.708 |
| search + oracle (diagnostic) | 6 / 0 / 2 | 0.750 | 0.708 |

Drop the mirror stratum and the production ordering reverses: agent1c goes from last to first and finishes level with the oracle, while the margin over direct C1 shrinks. Keep it and agent1c is the only production arm that loses its head-to-head against the very player it is supposed to improve on. Eight games decide each of those readings; neither is a finding.

## 5. Cost and search behaviour

| arm | search calls | C1 fwd/move | fwd pos/s | s/move mean | median | p90 | move-change vs direct |
|---|---|---|---|---|---|---|---|
| direct accepted Phase 9 C1 | 0 | 1.0 | 558 | 0.002 | 0.002 | 0.002 | — |
| search + remaining_count | 2,835 | 876.3 | 2717 | 0.322 | 0.324 | 0.342 | 13.7% |
| search + original_phase11 | 3,080 | 869.0 | 2703 | 0.322 | 0.326 | 0.343 | 13.7% |
| search + agent1c | 2,742 | 861.5 | 2677 | 0.322 | 0.328 | 0.349 | 15.1% |
| search + oracle (diagnostic) | 2,403 | 54.6 | 1351 | 0.040 | 0.041 | 0.042 | 19.6% |

Move-change rate is against the arm's own root Phase 9 action, which the match-time probe pins to the accepted direct player's decision on the same position.

Move-change rate by opponent:

| arm | Phase 9 direct | Strategic | Tactical | Scout-rush |
|---|---|---|---|---|
| direct accepted Phase 9 C1 | — | — | — | — |
| search + remaining_count | 10.1% | 10.5% | 9.0% | 18.0% |
| search + original_phase11 | 10.2% | 10.2% | 12.3% | 18.5% |
| search + agent1c | 12.1% | 11.5% | 18.5% | 18.6% |
| search + oracle (diagnostic) | 13.3% | 18.3% | 17.3% | 27.8% |

Strength bought per unit of search time:

| arm | EWR vs direct | extra s/move | extra s/game | EWR per extra search second (per game) |
|---|---|---|---|---|
| search + remaining_count | +0.1406 | 0.321 | 28.4 | +0.00496 |
| search + original_phase11 | +0.1406 | 0.320 | 30.7 | +0.00457 |
| search + agent1c | +0.1094 | 0.320 | 27.4 | +0.00400 |
| search + oracle (diagnostic) | +0.2031 | 0.039 | 2.8 | +0.07179 |

How the games ended:

| arm | battleless_move_limit_draw | flag_capture | opponent_no_legal_move |
|---|---|---|---|
| direct accepted Phase 9 C1 | 1 | 31 | 0 |
| search + remaining_count | 0 | 32 | 0 |
| search + original_phase11 | 0 | 31 | 1 |
| search + agent1c | 0 | 32 | 0 |
| search + oracle (diagnostic) | 0 | 32 | 0 |

Game length:

| arm | mean plies | median plies | player decisions | mean game s |
|---|---|---|---|---|
| direct accepted Phase 9 C1 | 227.8 | 146.0 | 3,644 | 0.3 |
| search + remaining_count | 177.0 | 131.0 | 2,835 | 28.8 |
| search + original_phase11 | 192.3 | 147.0 | 3,080 | 31.2 |
| search + agent1c | 171.2 | 112.5 | 2,742 | 27.8 |
| search + oracle (diagnostic) | 150.0 | 116.5 | 2,403 | 3.1 |

## 6. Match-time boundary probe

Each seat was re-asked a sample of its own decisions on a state whose hidden opponent identities had been permuted by the accepted `permute_hidden_identities`, and required to answer identically; the search seats were additionally required to agree with the accepted direct player on what the direct Phase 9 action was.

| arm | permutation checks | assignments actually changed | answer changed | direct-agreement checks | failures |
|---|---|---|---|---|---|
| direct accepted Phase 9 C1 | 16 | 16 | 0 | 0 | 0 |
| search + remaining_count | 16 | 16 | 0 | 16 | 0 |
| search + original_phase11 | 16 | 16 | 0 | 16 | 0 |
| search + agent1c | 16 | 16 | 0 | 16 | 0 |
| search + oracle (diagnostic) | 16 | 16 | 2 | 16 | 0 |

The oracle arm is the positive control: it reads the true world by design, and it changed its answer under permutation in 2 of its 16 checks. That is what makes the production arms' zero a result rather than a probe with no power — though a control that fires 2 times in 16 is weak evidence taken alone: a search decision is often robust to which world it sees, which is exactly why the structural boundary in the engine, not this probe, is what the anti-leak claim rests on.

## 7. Interpretation

The instruction's second branch applies. Search at SMALL — 16 worlds, up to 8 root candidates, 6 rollout plies — beat the direct accepted Phase 9 player with every belief provider tried, by +0.1094 to +0.1406 EWR, and the oracle arm beat it by more. That is a working search, not a search that needs a bigger budget to justify itself, and no world count or depth was raised to obtain it. The direction survives every slice of the set — head to head against the direct player itself the search arms took 14 of 24 games, and against the three rule opponents every search arm again finished above direct C1 — but the size of the margin does not: it ranges from a couple of games to five depending on which boards are counted.

On the instruction's own test — does Agent1C search beat direct C1 — the answer here is yes: 0.6250 against 0.5156 (+0.1094, 3.5 games of 32), paired at 4 boards better and 1 worse. So the configuration is preserved for Agent 4. What the same table does not support is a claim that Agent1C beliefs are the reason: agent1c placed 3 of 3 among the production arms on the full set, behind a count-based baseline that carries no learned belief at all — and first of three once the mirror stratum is removed. A configuration that changes rank with the slice is preserved because it is the phase's candidate and it did not lose, not because this set showed it to be the best one.

Perfect hidden information is still worth something to this search: the oracle arm finished +0.0625 EWR above the best production arm (2.0 games of 32), on 1 world instead of 16 and at an eighth of the latency. That gap is the headroom a better belief could in principle recover — and at this sample size it is itself inside the noise, so it sizes a direction to look, not a quantity to trust.

Production arm ordering by EWR: search + remaining_count 0.6562 > search + original_phase11 0.6562 > search + agent1c 0.6250. The whole spread is 1.0 game, against an unpaired standard error of about 0.085, so the ordering is a record of what happened and not a ranking. Notably it does not reproduce the belief-quality ordering: `remaining_count` has the worst beliefs of the three by construction and finished level with the best.

What search costs here is not small: 0.322 s/move against 0.0018 s/move, roughly 180x the per-move compute and about 28 extra seconds per game, for roughly a tenth of a point of EWR. Search also shortens games: 228 mean plies for direct C1 against 177, 192, 171 for the search arms and 150 for the oracle. Every search arm resolves a game sooner than the direct player does; the three production arms do not order among themselves by belief quality, so this is a search-versus-no-search effect and not a belief effect.

## 8. Limitations

- 32 games per arm over four opponents is an engineering sample, not a powered experiment: an EWR difference below the stated noise scale is not evidence of an ordering.
- One budget (SMALL), one beta, one candidate rule, one search version. No tuning was attempted and none is implied by the result.
- The arms share boards, opponent seeds and per-ply search seeds, which removes setup variance but leaves the arms correlated: paired numbers and unpaired numbers must not be mixed.
- The oracle arm is an offline diagnostic upper bound on the search mechanism, never a playable configuration, and its latency is not comparable to the belief arms' (it collapses to one world).
- The match driver holds the true engine state so the search seat can materialize worlds; the boundary is enforced structurally by the Agent 1 engine and checked at run time by the permutation probe, not by the policy-input isolation the accepted match runner provides.
- Setups come from the accepted library's 'validation' split, the same pool Phase 11B's dev split drew from, so a mild optimistic residual for agent1c is accepted for an engineering match test.

## 9. Deliverables and status

```text
stratego/search/phase12/matchplay.py           (new; Agent 1 and 2 modules untouched)
tests/search/test_phase12_matchplay.py
reports/phase12/agent_03_match_config.json
reports/phase12/agent_03_games.jsonl
reports/phase12/agent_03_games.csv
reports/phase12/agent_03_report.md
reports/phase12/agent_03_summary.json

phase11_final_classification     FAIL
phase11b_selection               Agent1C
scientific_validation_status     not performed
oracle_available_in_production   False
phase11_test_bank_used           False
search_core_modified             False
budget_above_small_used          False
budget_changed_during_run        False
agent_4_launched                 False
```

Stop condition reached: the compact match test is complete. No budget above SMALL was run, no world count or depth was raised to compensate, and Agent 4 is not launched.
