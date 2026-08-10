# Phase 4 Sequential Agent Plan

## Goal

Build a permanent, reproducible evaluation system for all later Stratego models.

Phase 4 should answer:

- Can we evaluate policies without hidden-information leakage?
- Can matches be reproduced exactly?
- Can evaluation run in parallel without changing results?
- Do we have baseline opponents of meaningfully different strength?
- Can we measure effective win rate with uncertainty?
- Can we diagnose results by color, setup, terminal reason, and opponent?
- Do we have stress opponents that expose brittle play?

## Agent sequence

| Agent | Task |
|---|---|
| 1 | Evaluation contracts, policy interface, match identity, fixed evaluation setup bank |
| 2 | Baseline and stress opponent suite |
| 3 | Parallel match runner, statistics, reproducibility, reporting |
| 4 | Hidden-information audit, baseline league calibration, final Phase 4 acceptance |

Run strictly in order.

## Shared report

```text
reports/phase_4_implementation_report.md
```

Owned sections:

```text
# Phase 4 Implementation Report

## 1. Agent 1 — Evaluation Foundations and Setup Bank
## 2. Agent 2 — Baseline and Stress Opponents
## 3. Agent 3 — Match Runner and Statistics
## 4. Agent 4 — Calibration, Security Audit, and Phase 4 Acceptance
```

Agent 1 creates the report header. Later agents append only their own section.

## Canonical data files

```text
reports/phase_4_data/agent_01_evaluation_foundations.json
reports/phase_4_data/agent_02_baseline_agents.json
reports/phase_4_data/agent_03_match_runner_statistics.json
reports/phase_4_data/agent_04_calibration_security.json
```

Optional raw CSV/JSON files may be added using the same `agent_0X_` prefix.

## Phase 4 target opponent categories

The phase should finish with at least:

1. random legal;
2. basic heuristic;
3. tactical rule-based;
4. strategic rule-based;
5. several unusual/stress policies.

The first four should form at least **three statistically distinguishable strength tiers** after calibration. Stress policies need not be stronger; they must produce meaningfully different behavior.

## Primary evaluation metric

Effective win rate:

\[
\mathrm{EWR} = rac{W + 0.5D}{N}.
\]

Reports must also preserve win/draw/loss separately.

For paired color/setup evaluation, confidence intervals should be computed using
the **paired match unit** rather than pretending all individual games are independent.

## Evaluation setup bank

Phase 4 uses a fixed deterministic evaluation-only setup bank.

It is **not** the Phase 7 training setup generator.

Recommended target:

- 512–1,024 deterministic setup pairs;
- legal and reproducible;
- versioned;
- reusable by future checkpoints.

## Global acceptance

Phase 4 is complete only if:

- every policy uses observer-legal information only;
- every policy always returns legal actions;
- fixed match identities reproduce exactly;
- parallel execution does not alter deterministic results;
- paired color evaluation works;
- effective win rate and confidence intervals are validated;
- complete game/replay references exist for diagnosis;
- >=100,000 policy-level hidden-state permutation checks produce zero unexplained policy differences;
- baseline calibration yields at least three useful strength tiers;
- stress agents create measurably different behavior;
- a machine-readable calibration report exists.
