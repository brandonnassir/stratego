# Phase 9 — Agent 6
# Bounded RL Pilot Selection

## Mission

Run exactly the six predeclared Phase 9 pilot configurations, under identical logical schedules, and freeze one `phase9_train_config_v1` using validation only.

This agent may select hyperparameters only from the frozen Agent 1 matrix.

No final-test inference is permitted.

## Prerequisites

Require Agents 1–5 `PASS` and formal acceptance.

Verify trainer/target/scheduler/population contract digests.

### Mandatory corpus resolver check

Resolve the accepted Phase 8 corpus through `synthetic_corpus.default_corpus_root()` and verify all accepted digests. No hard-coded absolute path in pilot/trainer/checkpoint code.

## Candidates

Exactly:

```text
P9-A   LR 1e-4   beta0 0.005
P9-B   LR 1e-4   beta0 0.020
P9-C   LR 3e-4   beta0 0.005
P9-D   LR 3e-4   beta0 0.020
P9-E   LR 6e-4   beta0 0.005
P9-F   LR 6e-4   beta0 0.020
```

No seventh candidate. No unregistered config.

## Fresh start fairness

Every candidate must:

- start from the exact Phase 8 accepted checkpoint;
- verify identical starting model-state checksum;
- start with fresh optimizer/scheduler/KL-controller state;
- use the same logical schedule identities;
- use the same setup/opponent assignments;
- use the same validation bank;
- run exactly 8 RL iterations;
- collect exactly 1,024 scheduled games per iteration;
- use 2 optimizer epochs per sealed rollout.

Different candidates naturally generate different trajectories after their policies diverge. Fairness means the **logical schedule**, not identical action bytes.

## Pilot schedule

Per iteration:

```text
current       512
historical    256
rule          154
stress        102
```

Rule:

```text
Strategic 77
Tactical  54
Basic     23
```

Historical semantics during pilots must follow Agent 1's frozen pilot rule. If pilot-local archives are permitted, all candidates must archive at identical logical points and use identical historical-selection rules.

## Hard vetoes

Veto if any frozen condition occurs:

```text
illegal neural action                   > 0
non-finite loss                         > 0
non-finite gradient                     > 0
non-finite parameter                    > 0
behavior identity mismatch              > 0
target reconstruction mismatch          > 0
observer-safety failure                 > 0
checkpoint/resume failure               > 0
mean iteration/epoch KL                 > 0.08
iteration PPO clip fraction             > 0.75
validation Random EWR                   < 0.90
validation Basic EWR                    < 0.60
```

A vetoed candidate receives no rescue rerun.

## Validation

Use `phase9_validation_bank_v1` only.

At the exact cadence frozen by Agent 1, evaluate greedy single-request float32 C1.

Selection score:

\[
S=0.45E_\text{Strategic}+0.35E_\text{Tactical}+0.20E_\text{Phase8-anchor}
\]

Higher is better.

Tie-break:

```text
higher score
higher Strategic EWR
lower mean behavior KL
higher examples/s
```

Random and Basic only gate regressions; they are not score components.

Do not use stress results for selection unless Agent 1 explicitly froze them as a tie-break.

## Winner

Select exactly one candidate.

Freeze:

```text
phase9_train_config_v1
```

Include:

```text
C1 identity
Phase 8 starting checkpoint SHA
learning rate
initial KL beta
all fixed PPO/value/belief parameters
entropy schedule
optimizer
batch size
epochs
population version
schedule version
historical archive rule
validation cadence
archive cadence
canonical iteration count
canonical games/iteration
loader/collector topology
all seeds
checkpoint version
acceptance versions
```

Hash the full document.

Also preserve a distinct trainer runtime identity if implementation uses a narrower runtime object; label namespaces explicitly as learned in Phase 8.

## No checkpoint continuation

Do not hand any trained pilot checkpoint to Agent 7.

Agent 7 must start freshly from the accepted Phase 8 anchor with the winning configuration.

Handoff includes only:

- configuration;
- digests;
- seeds;
- expected starting checksum;
- pilot evidence.

## Access instrumentation

Measure, not merely assert:

```text
Phase9 final-test neural games              0
Phase9 final-test neural checkpoint loads   0 if applicable
```

Keep an explicit access log.

## Artifacts

Create:

```text
reports/phase_9_data/agent_06_pilot_selection.json
reports/phase_9_data/agent_06_pilot_runs.csv
reports/phase_9_data/agent_06_frozen_train_config.json
```

The CSV should contain every validation checkpoint of every candidate and all score components.

## Completion gates

Require:

```text
agents1_5_pass
corpus_resolver_verified
corpus_digests_match
candidate_count_6
unregistered_candidates_0
identical_starting_checkpoint_identity
logical_schedule_fairness_pass
equal_iteration_budget
equal_game_budget
equal_validation_schedule
hard_veto_logic_exact
selection_score_reproducible
winner_unique
frozen_train_config_complete
frozen_train_config_digest_written
no_pilot_checkpoint_handed_forward
final_test_neural_access_zero
full_suite_green
```

## Forbidden

Do not:

- add a candidate;
- rerun a weak candidate with a new seed;
- change validation score;
- change population mix;
- change final budget from 60 iterations;
- use final-test results;
- use Phase 8 synthetic test policy imitation metrics for selection;
- begin canonical Agent 7 training.

## Handoff to Agent 7

Provide:

- winning candidate ID;
- complete `phase9_train_config_v1`;
- config document digest;
- runtime identity digest if distinct;
- exact fresh Phase 8 starting checkpoint SHA;
- expected model checksum;
- all Phase 9 seeds;
- population/archive contracts;
- validated collector/trainer topology;
- canonical budget and validation/archive cadence.

Agent 7 runs one fresh canonical job only.
