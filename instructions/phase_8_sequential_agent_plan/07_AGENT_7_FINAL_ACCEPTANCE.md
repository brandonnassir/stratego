# Phase 8 Agent 7 — Independent Held-Out Evaluation and Phase 8 Freeze

## Role

You are **Agent 7**, final Phase 8 validation agent.

You do not train or tune the model.

You open the sealed test corpus and frozen Phase 4 neural evaluation only after verifying Agent 6's checkpoint is immutable.

You may recommend `PASS`, `FAIL`, or `BLOCKED`.

The reviewing chat formally accepts or rejects Phase 8.

Do not begin Phase 9.

## Prerequisite

Agents 1–6 must all be PASS.

Read every Phase 8 artifact and the complete report.

Verify the frozen checkpoint SHA-256 before any evaluation.

Run the full suite before Agent 7 files.

## Independent checkpoint load

Rebuild C1 from config.

Load the Agent 6 checkpoint through the normal checkpoint API.

Require:

```text
model contract match
C1 digest match
parameter count 863,959
all parameters finite
checkpoint SHA-256 exact
corpus/example/train-config identities exact
```

Do not modify it.

## Sealed synthetic test evaluation

Now open the `test` selected-example universe.

Compute model and baseline metrics from scratch.

Primary metrics:

```text
policy:
    CE
    CE ratio to uniform legal
    top-1
    expected uniform top-1
    legal normalized entropy
    max-probability distribution

value:
    CE
    CE ratio to train-fitted prior
    Brier
    baseline Brier
    accuracy
    confusion matrix

belief:
    hidden-only CE
    CE ratio to remaining-count marginal
    top-1
    baseline top-1
    counts by true type
    metrics by true type
    metrics by progress bucket
```

Bootstrap confidence intervals by game identity.

Do not refit the value baseline on test.

## Test threshold gates

Require:

```text
policy CE ratio <= 0.90
policy top-1 > uniform expected top-1

value CE ratio <= 0.98
value Brier < baseline Brier

belief CE ratio <= 0.98
belief top-1 > baseline top-1

non-finite logits = 0
fraction legal max probability > 0.999 < 0.95
```

A miss is a real Phase 8 failure unless review changes the contract version.

Do not tune the checkpoint after seeing test results.

## Frozen Phase 4 random evaluation

Use the accepted evaluation harness, float32 `single_request`, frozen:

```text
evaluation_setup_bank_v1
1,024 setup pairs
color_swap_same_board
random baseline
```

Play the full:

```text
2,048 games
```

Require:

```text
EWR >= 0.950
Red EWR >= 0.900
Blue EWR >= 0.900
paired-bootstrap 95% lower bound > 0.900

illegal actions = 0
model failures = 0
non-finite outputs = 0
```

Do not modify the evaluation bank.

## Final versus canonical initialization

Evaluate the final checkpoint against the frozen untrained canonical C1 checkpoint.

Use at least:

```text
512 paired cases
1,024 total games
```

with accepted balanced-color semantics.

Require:

```text
final EWR >= 0.700
paired 95% lower bound > 0.550
```

This is not a hyperparameter decision; the checkpoint is already frozen.

## Additional baseline evaluations

Report but do not use them to rescue/fail Phase 8 unless a frozen hard gate covers them:

```text
Basic
Tactical
Strategic
stress policies where meaningful
```

Use at least 256 paired setup cases per reported tier when practical.

Also break down color, terminal reason, game length, and Phase 4 setup subgroup if present.

## Family-stratified synthetic test metrics

Using test setup provenance, report policy/value/belief metrics by the 16 Phase 7 primary setup families.

These are diagnostics.

Do not revise family sampling after seeing them.

## Independent upstream integrity review

Recompute:

```text
Phase 4 bank digest
Phase 7 library digest
C1 config digest
corpus manifest/content digests
test selected-example digest
checkpoint digest
```

Require all match prior accepted values.

## Training-discipline audit

From logs/artifacts prove:

```text
pilot used train+validation only
Agent 6 used train+validation only
test model inference before Agent 7 = 0
Phase 4 neural strength games before Agent 7 selection phase = 0
final checkpoint selected by validation only
final run started from canonical fresh initialization
candidate count <=6
```

This is a hard gate.

## Full repository suite

After writing Agent 7 artifacts, run the complete suite.

Artifact-gated tests must execute.

Record the steady-state result if the acceptance artifact contains the suite result and therefore requires a second non-circular run.

## Final freeze if PASS

Expected Phase 8 frozen identities:

```text
warmstart_training_contract_v1
synthetic_warmstart_corpus_v1
warmstart_decision_sampler_v1
warmstart_example_v1
warmstart_trainer_v1
warmstart_checkpoint_v1
warmstart_train_config_v1
warmstart_eval_v1
```

Freeze:

```text
accepted checkpoint path/digest
C1 architecture digest
training config
corpus digests
baseline definitions
test results
Phase 4 random evaluation
```

Later semantic changes require a new version.

## Known limitations to carry forward

At minimum discuss:

- Phase 8 learns by imitation/outcomes from rule agents, not self-play;
- stress/random decisions do not supervise policy;
- value labels are final outcomes rather than RL advantages;
- belief is a lightweight shared head and Phase 11 still owns deeper belief validation;
- no learned setup policy;
- no search;
- no dynamic damping;
- synthetic-teacher biases;
- any corpus/trainer throughput bottleneck;
- any MPS resume numerical tolerance;
- any remaining crash/recovery limitation.

## Required artifacts

Create:

```text
reports/phase_8_data/agent_07_heldout_metrics.json
reports/phase_8_data/agent_07_random_evaluation.json
reports/phase_8_data/agent_07_final_acceptance.json
reports/phase_8_data/agent_07_phase9_handoff.json
```

Append report section 7.

## PASS gates

Recommend PASS only if every global common-contract Phase 8 gate passes.

Do not average a failed belief/value gate into a good playing-strength score.

Do not ignore a random EWR miss because another baseline was beaten.

Do not retrain after opening test.

## Final handoff format

Return:

```text
Phase 8 recommendation

frozen checkpoint
checkpoint SHA-256
C1 parameter count/config digest

corpus version/digests
train/validation/test games
selected train/validation/test examples

policy test CE ratio / top-1
value test CE ratio / Brier
belief test CE ratio / top-1
entropy/collapse result

Random evaluation:
    games
    W/D/L
    EWR
    Red EWR
    Blue EWR
    paired CI

final vs initial:
    games
    EWR
    paired CI

additional baseline diagnostics

checkpoint/resume status
corpus crash/reconcile status
observer-safety status
Phase 4 bank integrity
Phase 7 library integrity

tests before/after
known limitations

Phase 9:
READY TO PLAN
or
BLOCKED
```

Stop.

Do not begin Phase 9 self-play.
