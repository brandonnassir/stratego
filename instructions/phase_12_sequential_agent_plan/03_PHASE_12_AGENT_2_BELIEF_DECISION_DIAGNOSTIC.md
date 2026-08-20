# Phase 12 — Agent 2
## Belief-to-Decision Diagnostic

## Mission

Determine whether better beliefs actually cause the same search algorithm to choose better-looking actions.

Primary question:

> Does Agent 1C improve search decisions relative to remaining-count and the original Phase 11 neural belief?

Do not run a large tournament.

## 1. Prerequisite

Start from the accepted Agent 1 search core.

Do not alter search algorithm, root candidate rule, Phase 9 policy/value, belief models, or prior artifacts.

Only the belief provider changes across the main search arms.

## 2. Fresh diagnostic position set

Create approximately:

```text
256 positions

64 Phase9-like
64 Strategic
64 Tactical
64 Scout-rush
```

Do not use the spent Phase 11 sealed test bank.

Prefer positions with meaningful unresolved opponent information.

Use balanced colors where practical.

## 3. Search budget

Run TINY first.

If latency is clearly acceptable and implementation stable, use SMALL as the primary comparison.

Do not scale beyond SMALL.

## 4. Compare

Evaluate:

```text
direct accepted Phase 9 C1
search + remaining_count
search + original_phase11
search + agent1c
search + oracle
```

For every search arm keep identical:

```text
Phase 9 policy/value
candidate-action rule
world count
search depth
rollout policy
policy regularization
seed policy
```

Only the belief provider changes.

## 5. Diagnostics

For each arm report:

```text
move disagreement rate vs direct C1
agreement rate with oracle-search action
mean/median search score delta vs direct action
move latency
results by behavior group
```

Also report pairwise action agreement among:

```text
remaining_count
original_phase11
agent1c
oracle
```

## 6. Oracle use

Oracle search may use true hidden state only as an offline diagnostic.

Use it to answer:

> If this same search mechanism had perfect hidden information, what move would it prefer?

Oracle alignment is diagnostic, not proof of optimality.

## 7. Interpretation

Focus especially on Agent1C search vs original Phase 11 search.

If improved belief prediction does not change decisions, increase oracle agreement, or improve root scores, report that before more compute is spent.

If all belief variants are nearly identical, current search may be too shallow or insensitive. Do not immediately respond by increasing world count or depth in this agent.

## 8. Deliverables

Create:

```text
fresh diagnostic position manifest
reports/phase12/agent_02_report.md
reports/phase12/agent_02_summary.json
```

## 9. Stop condition

Stop after the position-level comparison.

Do not run a large match set.

Do not begin Agent 3 automatically.
