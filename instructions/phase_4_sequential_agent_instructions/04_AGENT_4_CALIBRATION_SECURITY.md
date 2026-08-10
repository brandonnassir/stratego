# Phase 4 Agent 4 — Baseline Calibration, Hidden-Information Audit, and Phase 4 Acceptance

## Role

You are **Agent 4** in a sequential Phase 4 implementation for the Stratego project.

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

Agents 1-3 must all be `PASS`.

Read the complete shared report and all prior Phase 4 JSON files.

## Objective

Perform the final Phase 4 acceptance work.

This agent owns:

- large policy-level hidden-information audit;
- baseline league calibration;
- strength-tier analysis;
- stress-agent behavior characterization;
- final reproducibility checks;
- final Phase 4 decision.

Do not begin neural model work or self-play training.

## Files you own

Suggested:

```text
scripts/run_phase4_agent04.py
tests/evaluation/test_phase4_acceptance.py
reports/phase_4_data/agent_04_calibration_security.json
reports/phase_4_data/agent_04_baseline_league_raw.csv
reports/phase_4_data/agent_04_hidden_information_audit.json
```

You may add a small calibration helper module if needed, but do not rewrite Agents 1-3.

## Full hidden-information audit

Across all baseline and stress policies:

1. generate or sample valid public positions;
2. clone the privileged state;
3. permute true identities among unresolved hidden opponent pieces while preserving public constraints;
4. build observer-safe policy inputs;
5. use identical policy seeds;
6. compare action and public diagnostics.

Acceptance target:

- at least **100,000 valid policy-level hidden-state permutation trials total**;
- every policy represented;
- multiple game phases represented;
- zero unexplained policy-action differences;
- zero unexplained public-diagnostic differences.

Also require a positive control proving the privileged hidden state/belief target actually changed.

Any leak is a hard FAIL.

## Baseline calibration league

Use Agent 3's runner and Agent 1's fixed setup bank.

Run a reproducible league containing at least:

```text
Random Legal
Basic Heuristic
Tactical Rule-Based
Strategic Rule-Based
all stress policies
```

### Screening

Use shorter schedules first to detect:

- broken policies;
- near-identical policies;
- pathological draw loops;
- obvious strength inversions caused by bugs.

If a policy is clearly misimplemented, fix the policy code within the intended design and rerun the relevant tests.

Do not redesign opponents merely to manufacture arbitrary numerical spacing.

### Final calibration

Use paired color/setup evaluation.

For the main ladder comparisons, use enough paired units to obtain stable confidence intervals.

Recommended target:

- **at least 512 paired units per important adjacent-tier comparison**;
- preferably 1,024 if runtime remains reasonable.

The exact league need not evaluate every policy against every other policy at the maximum sample size.

## Strength-tier gate

Do not require arbitrary fixed percentages.

Instead, Phase 4 must end with at least **three useful statistically distinguishable strength tiers** among the core ladder.

A practical interpretation:

- confidence intervals / paired comparisons provide evidence that at least three levels are meaningfully separated;
- Random should function as the floor;
- stronger baselines should not collapse into statistically indistinguishable copies of one another.

If Basic/Tactical/Strategic are too similar, modestly revise heuristic weights/rules and recalibrate, but preserve their conceptual roles.

Document every tuning iteration.

## Stress-policy characterization

Stress policies need not fit the strength ladder.

Show they create meaningfully different behavior using metrics such as:

- attack rate;
- Scout move frequency;
- Miner move/attack frequency;
- average game length;
- battleless draw rate;
- Flag-capture rate;
- reveal/combat frequency;
- movement entropy / action-distribution diversity where practical.

At least several stress policies should differ materially from the core Strategic baseline on one or more relevant metrics.

## Final reproducibility check

Take a representative subset of the final league and rerun it:

- serially;
- with the normal parallel worker count;
- with match order shuffled.

Require identical per-match results and replay digests.

## Final Phase 4 report

Append:

```markdown
## 4. Agent 4 — Calibration, Security Audit, and Phase 4 Acceptance
```

End with an explicit block:

```text
Phase 4 decision: PASS / FAIL
Baseline tiers established: <count>
Hidden-information audit trials: <count>
Hidden-information mismatches: <count>
Evaluation harness ready for future checkpoints: yes/no
```

## Data file

Primary:

```text
reports/phase_4_data/agent_04_calibration_security.json
```

Minimum:

```text
agent
status
hidden_information_trials
hidden_information_mismatches
positive_control_trials
positive_control_failures
policies_audited
league_match_count
league_paired_unit_count
core_policy_results
pairwise_effective_win_rates
pairwise_confidence_intervals
strength_tier_count
strength_tier_membership
stress_behavior_metrics
reproducibility_rerun_matches
reproducibility_mismatches
policy_tuning_iterations
phase4_decision
test_total
test_passed
test_failed
files_created
files_modified
```

## Final Phase 4 completion gate

PASS only if:

- all prior agents passed;
- >=100,000 hidden-state policy permutation trials produce zero unexplained differences;
- positive controls verify privileged state really changed;
- every baseline/stress policy always produces legal actions;
- final league is reproducible;
- paired confidence intervals work;
- at least three useful core strength tiers exist;
- stress policies create meaningfully distinct behavior;
- raw results permit game reproduction;
- the full repository test suite is green.

If all pass:

```text
Phase 4 = COMPLETE
```

Do not modify `stratego_project_docs/`.

The project-document update will be performed after the Phase 4 report is reviewed and accepted.
