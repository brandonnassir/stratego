# Phase 9 — Agent 2
# Population and Opponent Scheduler

## Mission

Implement and exhaustively audit the deterministic **logical population schedule** for Phase 9.

This agent decides **which logical games should exist**, not how neural moves are collected and not how RL optimization works.

Read the common contract and Agent 1 artifacts first.

## Prerequisites

Require Agent 1 `PASS` and formal reviewer acceptance.

Verify all Agent 1 contract and bank digests from live source.

Verify frozen upstream versions and Phase 8 accepted checkpoint identity.

### Mandatory corpus resolver check

Call:

```text
synthetic_corpus.default_corpus_root()
```

Require the accepted current path and all three accepted corpus digests. Do not hard-code the absolute path into scheduler/library code. Digest mismatch => `BLOCKED`.

## Implement

Create a versioned deterministic population/schedule layer implementing:

```text
phase9_population_v1
phase9_rollout_schedule_v1
```

The schedule must be a pure function of:

- rollout version;
- Phase 9 master/schedule/opponent seeds;
- run namespace (`pilot candidate` or `canonical`);
- RL iteration;
- logical game ordinal;
- frozen active historical archive identities.

Worker count, process partitioning, enumeration order, arrival order, wall clock, and resume boundary must not affect logical game identity.

## Canonical schedule

Exactly 2,048 games per iteration:

```text
current/current       1,024
current/historical      512
current/rule            307
current/stress          205
```

Rule games:

```text
Strategic 154
Tactical  107
Basic      46
```

Pilot schedule exactly 1,024:

```text
current       512
historical    256
rule          154
stress        102
```

Pilot rule games:

```text
Strategic 77
Tactical  54
Basic     23
```

## Current/current identity

Both players use the exact same immutable behavior snapshot identity for that iteration.

```text
learner_control = both
```

Do not treat one side as an opponent for training eligibility.

## Historical identity

Initial historical archive:

```text
H000 = Phase 8 accepted checkpoint
```

Canonical archive schedule:

- new immutable archive snapshot eligible every 5 completed RL iterations;
- active history = Phase 8 anchor + 8 most recent eligible snapshots;
- historical game assignment uniform over active identities under deterministic hashing.

The scheduler must accept the current active archive manifest as explicit immutable input. It must never inspect match outcomes to select an opponent in `phase9_population_v1`.

## Rule and stress identities

Use exact frozen Phase 4 policy tokens and versions.

Stress bucket must deterministically allocate across the six accepted stress policies with long-run count imbalance no larger than one game when mathematically possible.

Do not modify rule/stress policies.

## Color balance

For every asymmetric bucket:

- current learner appears approximately equally as Red and Blue;
- exact counts follow Agent 1's frozen parity rule;
- odd remainder alternates by iteration parity;
- independent reconstruction reproduces color assignment exactly.

## Setup assignment

Use the frozen Phase 7 training setup source and `neutral_v1`.

Setup assignment must be identity-derived and independent of worker/process order.

Do not outcome-weight setup families.

No Phase 9 validation/test setup may enter train rollouts.

## Game record

Each logical scheduled game must expose at least:

```text
phase9_game_id
run_namespace
rl_iteration
game_ordinal
bucket
red_policy_identity
blue_policy_identity
learner_control
behavior_snapshot_identity
historical_snapshot_identity_if_any
setup_identity_seed/root
red_setup_source_identity
blue_setup_source_identity
policy RNG identities
```

Do not include privileged piece truth in scheduler records.

## Exhaustive schedule audits

Before Agent 3:

1. enumerate all pilot schedules for six candidates;
2. enumerate the full 60-iteration canonical logical schedule;
3. prove no duplicate game IDs;
4. prove no cross-namespace collisions;
5. prove exact bucket counts every iteration;
6. prove rule subdivisions;
7. prove stress allocation;
8. prove color balance;
9. prove setup split is train only;
10. prove all 16 setup families receive broad coverage;
11. prove worker/order independence;
12. prove resume = scheduled minus completed IDs.

Also explicitly audit seed collisions separately for any finite-width derived seed namespaces used for:

```text
setup
red neural/rule decisions
blue neural/rule decisions
opponent assignment
historical assignment
```

Any collision that violates the frozen uniqueness contract is `BLOCKED`.

## No neural semantics

You may construct/load checkpoint identities to verify hashes, but do not run meaningful neural self-play or training.

Tiny scheduler smoke tests may use mocked/dummy policy identities.

## Artifacts

Create:

```text
reports/phase_9_data/agent_02_population.json
reports/phase_9_data/agent_02_schedule_audit.json
reports/phase_9_data/agent_02_canonical_schedule_summary.csv
```

The CSV should summarize iteration × bucket × opponent × learner color counts.

## Completion gates

Require:

```text
agent1_pass
contract_digests_match
corpus_resolver_verified
corpus_digests_match
pilot_schedules_exact
canonical_60_iteration_schedule_exact
canonical_total_games_122880
duplicate_game_ids_zero
seed_collision_violations_zero
bucket_count_mismatches_zero
rule_subdivision_mismatches_zero
stress_allocation_mismatches_zero
color_balance_violations_zero
train_setup_split_violations_zero
worker_order_dependence_zero
resume_identity_mismatches_zero
no_neural_training
full_suite_green
```

## Forbidden

Do not:

- implement PPO;
- choose behavior probability representation differently from Agent 1;
- change population weights;
- change game counts;
- outcome-prioritize historical opponents;
- train learned setup selection;
- open final-test evaluation;
- run the Phase 9 canonical training job.

## Handoff to Agent 3

Provide:

- schedule enumeration API;
- pure game-ID parser/rebuilder;
- active-history manifest interface;
- learner-control field;
- exact policy/checkpoint identities;
- exact setup identity derivation;
- resume subtraction semantics;
- schedule digests and audits.

Agent 3 must collect exactly these games.
