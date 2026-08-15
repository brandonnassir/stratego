# Phase 8 Agent 1 — Warm-Start Contract and Pre-Training Acceptance Standard

## Role

You are **Agent 1** of sequential Phase 8.

Your job is to freeze the Phase 8 experiment contract **before any synthetic production corpus is generated and before any meaningful optimizer step is taken**.

Do not build the production corpus. Do not run pilot training. Do not begin Agent 2.

## Required reading

Read:

```text
00_PHASE_8_SEQUENCE_AND_COMMON_CONTRACT.md

reports/phase_7_implementation_report.md
reports/phase_7_data/
data/setups/setup_library_v1_manifest.json

reports/phase_6_implementation_report.md
reports/phase_6_data/

reports/phase_4_implementation_report.md
reports/phase_4_data/

stratego/evaluation/
stratego/training/
stratego/model/
stratego/setups/

stratego_project_docs/05_project_plan.md
stratego_project_docs/09_public_event_and_replay_schema.md
```

Use actual repository paths if the docs are elsewhere.

Run the full suite before edits using the accepted project virtual environment.

## Frozen upstream verification

Verify and record:

```text
stratego_project_v1
phase2_1_reference_1.2.0
observation_v2_1_127ch
model_contract_v2
C1 = 863,959 parameters
C1 config digest 31ca84ab...e07d
trajectory_v1
evaluation_setup_bank_v1
Phase 4 bank digest 5fe5f987...674266

setup_library_v1
library digest 7b8a6660...02777
setup_sampler_v1
setup_perturbation_v1 / seed_encoding_v1
setup_source_v1
neutral_v1
```

Stop if a frozen value disagrees.

## Mission

Define and serialize:

```text
warmstart_training_contract_v1
synthetic_warmstart_corpus_v1
warmstart_decision_sampler_v1
warmstart_example_v1
warmstart_eval_v1
```

and freeze:

1. exact Phase 4 rule-policy roster;
2. exact policy-supervision weights;
3. exact ordered matchup schedule;
4. train/validation/test setup-source semantics;
5. deterministic game/seeding identity;
6. decision-sampling contract;
7. policy/value/belief target semantics;
8. baseline semantics;
9. loss normalization semantics;
10. pilot candidate matrix and selection metric;
11. canonical Phase 8 seeds;
12. final acceptance thresholds;
13. test/Phase 4 bank sealing rules.

## Rule-agent roster

Read the live Phase 4 registry rather than inventing names.

Expected:

```text
strategic
tactical
basic
random
six accepted stress/unusual policies
```

Expected total:

```text
10 policies
```

Record for every policy:

```text
policy id
policy version
implementation path
deterministic/stochastic behavior contract
seed/tie-break semantics
role: tier or stress
policy-supervision weight
```

Do not change any Phase 4 policy.

If the live accepted roster is not the expected 10 policies, stop `BLOCKED` and report the discrepancy.

## Policy-supervision weights

Freeze:

```text
strategic    1.0
tactical     1.0
basic        0.5
random       0.0
all stress   0.0
```

If exact Phase 4 IDs differ in spelling/version, map those accepted IDs to the roles above.

Stress/random games remain fully eligible for value and belief supervision.

## Matchup schedule

For every one of the expected 100 ordered Red-policy / Blue-policy cells:

```text
train         200 games
validation     40 games
test           40 games
```

Require:

```text
train         20,000
validation     4,000
test           4,000
total         28,000
```

Do not let natural random sampling determine matchup counts.

## Setup sources

Freeze:

```text
train:
    training_setup_source('neutral_v1')

validation:
    audit/evaluation setup source
    split='validation'
    written justification:
        Phase 8 held-out warm-start validation corpus

test:
    audit/evaluation setup source
    split='test'
    written justification:
        Phase 8 sealed held-out warm-start test corpus
```

Each game samples Red/Blue setups independently.

No setup family gets outcome-based weighting.

## Corpus game identity

Specify stable, parseable identity.

Recommended conceptual fields:

```text
synthetic_warmstart_corpus_v1
split
red_policy_id
blue_policy_id
matchup_ordinal
```

Hash/seed derivation must also include a frozen corpus master seed.

Define separate domains for:

```text
setup identity / setup source root seed
red rule-policy randomness
blue rule-policy randomness
decision-sampler randomness
```

No global RNG cursor.

## Freeze Phase 8 seeds

Choose and record, before any result:

```text
corpus master seed
canonical C1 init seed
train shuffle/order seed
pilot namespace seed
final-run namespace seed
validation bootstrap seed
test bootstrap seed
```

Do not select a seed based on model results.

Use stable derivation functions rather than ad hoc arithmetic at call sites.

## Decision sampler

Freeze the common contract exactly:

```text
warmstart_decision_sampler_v1
max decisions/game = 64
```

Short game: all decisions.

Long game: 64 deterministic stratified bins, one seeded position per bin.

Specify exact bin-boundary arithmetic and tie behavior so Agent 3 can independently reproduce it.

## Example / target schema

Freeze exact field names/types for `warmstart_example_v1`.

The schema must include:

```text
observation
legal mask
acting player
absolute action
normalized model action
policy weight
WDL target
belief target
belief mask
game id
decision index
source policy
corpus split
```

Privileged metadata may be attached to the training example object but must never be part of the model-input object.

## Baselines

Freeze exact definitions from the common contract:

```text
policy       uniform legal
value        one constant WDL prior fitted on train selected examples only
belief       observable unresolved-inventory marginal
```

Specify:

```text
epsilon/log handling
tie-breaking for top-1 baseline
aggregation units
game-level bootstrap method
```

## Pilot candidate matrix

Publish no more than 6 candidates.

Keep fixed:

```text
C1 architecture
float32
batch size 256 unless your candidate matrix explicitly permits one Agent-4-proven alternative
AdamW optimizer family
gradient clipping required
same training corpus
same pilot selected-example universe
same model-init seed
same pilot update budget
```

Allowed candidate dimensions:

```text
learning rate
loss weights
weight decay
simple LR schedule choice
```

Recommended search shape:

```text
3 learning rates x 2 loss-weight profiles = 6
```

Do not include test performance, Phase 4 game strength, architecture, teacher weights, or setup sampling in the candidate matrix.

Record exact candidate IDs before Agent 5 can run.

## Pilot selection score

Freeze:

```text
r_policy = validation policy CE / policy baseline CE
r_value  = validation value CE  / value baseline CE
r_belief = validation belief CE / belief baseline CE

score = mean(r_policy, r_value, r_belief)
```

Lower is better.

Freeze hard veto and tie-break rules from the common contract.

## Final acceptance thresholds

Copy the common-contract Phase 8 gates verbatim into the machine-readable threshold artifact.

Do not relax them later.

Important final gates include:

```text
Random EWR >= 0.950 over 2,048 frozen Phase 4 games
both color EWRs >= 0.900
paired bootstrap lower bound > 0.900

final vs canonical initialization EWR >= 0.700 over >=1,024 games

test policy CE ratio <= 0.90
test value CE ratio <= 0.98 and Brier better
test belief CE ratio <= 0.98 and top-1 better
```

## Held-out sealing

Create an explicit access-policy utility or contract object if useful.

Before Agent 7:

```text
test:
    structural audit allowed
    model inference prohibited

Phase 4 bank:
    existing non-neural regression tests allowed
    neural playing-strength evaluation prohibited for selection
```

The enforcement should be testable.

Do not make it impossible for Agent 7 to open the resources after Agent 6 freezes the checkpoint.

## Suggested files

```text
stratego/training/warmstart_contract.py
stratego/training/warmstart_seed.py

tests/training/test_warmstart_contract.py
tests/training/test_warmstart_seed.py

scripts/run_phase8_agent01.py
```

Do not implement corpus generation/training yet.

## Required artifacts

Create:

```text
reports/phase_8_data/agent_01_warmstart_contract.json
reports/phase_8_data/agent_01_teacher_population.json
reports/phase_8_data/agent_01_acceptance_thresholds.json
```

Append only report section 1.

## PASS gates

PASS only if:

- Phase 7 formal acceptance verified;
- all upstream versions/digests match;
- exact 10-policy Phase 4 roster reproduced;
- 100 ordered matchup cells defined;
- exact 20k/4k/4k game schedule frozen;
- setup split semantics frozen;
- teacher policy weights frozen;
- corpus identity/seeds frozen;
- decision sampler exact;
- target semantics exact;
- baselines exact;
- pilot matrix <=6 and predeclared;
- pilot selection score exact;
- final acceptance thresholds frozen;
- test/Phase 4 selection restrictions explicit;
- no production corpus generated;
- no meaningful optimizer step run;
- full suite green.

## Handoff to Agent 2

Provide:

```text
all contract versions
exact policy roster and versions
policy weights
corpus seeds
game-id function
ordered matchup schedule
setup-source configuration
expected corpus storage schema
commit/reconciliation requirements
```

Agent 2 must not make new learning-design decisions.
