# Phase 4 Agent 3 — Parallel Match Runner, Statistics, and Reporting

## Role

You are **Agent 3** in a sequential Phase 4 implementation for the Stratego project.

Work from the **root of the Stratego repository**.

Complete only the work assigned in this file. Do not begin later-agent tasks.

## Frozen project contracts

Do not alter:

- reference engine: `phase2_1_reference_1.1.0`
- rules: `stratego_project_v1`
- observation: `observation_v2_1_127ch`
- action encoding: 10,000 source-destination action identifiers
- Phase 3 production simulator decision: `KEEP_PYTHON`

The Python reference engine remains the behavioral authority.

Do not modify game rules, observation semantics, action encoding, combat behavior,
terminal precedence, replay semantics, or hidden-information rules.

## Phase 4 scope

Phase 4 is:

> **Baseline opponents and evaluation harness**

It is **not**:

- neural-network training;
- final model architecture selection;
- setup-generator Phase 7 work;
- self-play reinforcement learning;
- belief-model training;
- decision-time search;
- browser/interface work;
- optimized-engine work.

The untrained Phase 3 representative Transformer is a benchmark probe and must
not be treated as a real Stratego opponent.

## Required reading

Before implementation, read at least:

```text
stratego_project_docs/README.md
stratego_project_docs/02_project_ruleset.md
stratego_project_docs/03_game_engine_spec.md
stratego_project_docs/04_engine_validation_plan.md
stratego_project_docs/05_project_plan.md
stratego_project_docs/06_observation_v2_127ch.md
stratego_project_docs/08_internal_state_spec.md
stratego_project_docs/09_public_event_and_replay_schema.md
reports/phase_3_implementation_report.md
```

Also inspect all prior code/tests your task depends on.

## Shared reporting contract

All Phase 4 agents append to one Markdown report:

```text
reports/phase_4_implementation_report.md
```

Each agent owns one numbered top-level section and must not rewrite earlier
sections.

All raw/machine-readable results go under:

```text
reports/phase_4_data/
```

Each agent must use its own filenames and never overwrite another agent's data.

Each report section must contain:

1. `PASS`, `FAIL`, or `BLOCKED`;
2. implementation summary;
3. files created/modified;
4. tests run;
5. measured results;
6. deviations/limitations;
7. data-file paths;
8. handoff notes.

## General correctness rules

- Policies may use only information legal for the acting player.
- Privileged `GameState` fields and belief targets must never enter a policy.
- Every returned action must be legal.
- The engine remains the final legality authority.
- Randomness must be reproducible from explicit seeds.
- Match identity must not depend on worker count or execution order.
- Hidden-information permutation tests must fail loudly on any unexplained policy change.
- Do not silently weaken tests to make a result pass.

## Stop conditions

Mark `BLOCKED` and stop if:

- a prerequisite agent did not pass;
- frozen engine semantics would need modification;
- policy code requires privileged hidden information;
- deterministic reproduction cannot be maintained;
- an evaluator cannot reproduce the same scheduled match from its identifiers;
- you discover an unresolved engine correctness issue.

Do not fix frozen-engine semantics yourself.

## Prerequisite

Agents 1 and 2 must both be `PASS`.

Read their report sections and data files.

## Objective

Build the reproducible evaluation engine that runs policies against one another and produces statistically correct summaries.

This agent owns:

- game/match execution;
- paired scheduling;
- parallel CPU execution;
- raw result schema;
- effective win rate;
- confidence intervals;
- league-rating utility;
- result aggregation/report generation.

Do not calibrate/tune the baseline ladder beyond small smoke runs.

## Files you own

Suggested layout:

```text
stratego/evaluation/match_runner.py
stratego/evaluation/scheduler.py
stratego/evaluation/statistics.py
stratego/evaluation/reporting.py
tests/evaluation/test_match_runner.py
tests/evaluation/test_statistics.py
tests/evaluation/test_parallel_reproducibility.py
scripts/run_phase4_agent03.py
reports/phase_4_data/agent_03_match_runner_statistics.json
```

Optional raw files:

```text
reports/phase_4_data/agent_03_reproducibility_raw.csv
```

## Match execution

A game runner must:

- create the frozen project game using the match specification;
- provide only observer-safe policy inputs;
- alternate policies correctly;
- reject any illegal policy action loudly;
- retain enough information for exact replay;
- report terminal result/reason;
- preserve setup IDs, colors, seeds, and policy versions.

Do not hide policy failures by substituting a random legal move.

## Parallel execution

Support parallel CPU evaluation.

Requirements:

- match identity is assigned before worker dispatch;
- worker number does not enter game randomness;
- result order may vary, but result content may not;
- rerunning the same schedule at different worker counts must produce identical per-match results.

Test at multiple worker counts, including serial execution.

## Raw match result

Minimum fields:

```text
match_id
paired_unit_id
candidate_policy_id/version
opponent_policy_id/version
candidate_color
setup_pair_id
replicate
root seed
policy seeds
winner
draw
candidate result
terminal reason
plies
replay/action-history reference or embedded compact replay
wall-clock timing
policy error field
```

## Statistics

Implement and test:

### Effective win rate

\[
\mathrm{EWR} = rac{W + 0.5D}{N}.
\]

Always also report:

```text
wins
draws
losses
```

### Paired confidence interval

Use the **paired evaluation unit** as the bootstrap/resampling unit.

Do not bootstrap individual games independently when they belong to a paired color/setup unit.

Implement a reproducible bootstrap confidence interval with an explicit seed.

Default recommendation:

- 95% interval;
- at least 10,000 bootstrap resamples for final reports;
- a smaller count may be used in unit tests.

### Color split

Report:

- candidate as Red;
- candidate as Blue.

### Other summaries

Report at least:

- mean/median plies;
- terminal-reason frequencies;
- setup-pair stratification;
- illegal/policy-error count;
- per-opponent summary.

## League rating

Provide a secondary rating utility such as Bradley-Terry / Elo-style estimation.

This is for convenient ranking only.

Do not replace effective win rate with Elo as the project success metric.

The method must be deterministic from raw results and documented.

## Reproducibility acceptance

Run a substantial schedule under at least:

```text
1 worker
2 workers
4 workers
8 workers
```

or the closest sensible values if platform/runtime limits intervene.

Require identical:

- match IDs;
- setups;
- action histories/replay digests;
- results;
- terminal reasons;
- plies.

Ordering of returned rows may differ.

## Statistical validation

Use synthetic known-result tables to test:

- all wins -> EWR 1.0;
- all losses -> 0.0;
- all draws -> 0.5;
- mixed outcomes;
- paired bootstrap reproducibility;
- color splits;
- missing/duplicate pair detection;
- aggregation invariant to row ordering.

## Data file

Write:

```text
reports/phase_4_data/agent_03_match_runner_statistics.json
```

Minimum:

```text
agent
status
match_result_schema_version
runner_version
statistics_version
matches_run
paired_units_run
worker_counts_tested
parallel_reproducibility_mismatches
illegal_policy_actions
policy_errors
statistics_unit_tests
bootstrap_resamples_acceptance
bootstrap_seed
league_method
test_total
test_passed
test_failed
files_created
files_modified
```

## Report section

Append only:

```markdown
## 3. Agent 3 — Match Runner and Statistics
```

## Completion gate

PASS only if:

- exact match reproduction works;
- parallel worker count does not change results;
- paired match scheduling works;
- EWR and 95% paired confidence intervals are implemented and tested;
- color/setup/terminal summaries work;
- raw match records are sufficient to reproduce a game;
- policy failures are loud and never converted into legal substitute actions.

Do not perform the final baseline calibration.
