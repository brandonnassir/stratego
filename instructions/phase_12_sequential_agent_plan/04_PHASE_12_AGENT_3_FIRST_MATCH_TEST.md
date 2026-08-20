# Phase 12 — Agent 3
## First Search Match Test

## Mission

Move from position diagnostics to the practical question:

> Does search actually make the player stronger?

This is a small engineering strength comparison, not a scientific final evaluation.

## 1. Arms

Compare:

```text
A. direct accepted Phase 9 C1
B. search + remaining_count
C. search + original_phase11
D. search + agent1c
```

Run an oracle-search arm only if inexpensive and it does not materially lengthen the experiment.

## 2. Search budget

Use the best working configuration from Agent 2.

Do not exceed SMALL:

```text
worlds <= 16
root moves <= 8
depth <= 6
```

Do not increase search budget during the run.

## 3. Opponents

Use:

```text
Phase 9 direct
Strategic
Tactical
Scout-rush
```

A reasonable engineering target is:

```text
8 balanced-color games per opponent per production search arm
```

Approximately 32 games per search arm.

This is not a statistical power calculation.

If wall-clock makes the full target excessive and the ordering is already clear enough to guide engineering, stop earlier and report exactly what completed.

Do not expand the experiment.

## 4. Match conditions

Keep common:

```text
setup source policy
color balancing
game rules
draw rules
search preset
Phase 9 move/value model
```

Search arms differ only in belief provider.

Do not use the Phase 11 sealed test bank.

## 5. Report

For every arm report:

```text
W / D / L
effective win rate
EWR vs each opponent
overall EWR
move latency
average game wall-clock
search calls
search-vs-direct move-change rate
```

Highlight:

```text
Agent1C search vs direct C1
Agent1C search vs old-belief search
Agent1C search vs remaining-count search
```

## 6. Decision logic

If all search variants are weaker than direct C1:

> Search mechanics are the likely problem.

Do not scale world count/depth just to compensate.

If Agent1C search beats direct C1 or appears clearly promising, preserve that configuration for Agent 4.

If oracle search was run and is also weak, treat that as strong evidence that search mechanics are the main bottleneck.

If oracle is strong but Agent1C is weak, belief quality or belief-to-world conversion may remain limiting.

## 7. Deliverables

Create:

```text
match configuration
match results
reports/phase12/agent_03_report.md
reports/phase12/agent_03_summary.json
```

## 8. Stop condition

Stop after this compact match test.

Do not increase to 32 or 64 worlds.

Do not begin Agent 4 automatically.
