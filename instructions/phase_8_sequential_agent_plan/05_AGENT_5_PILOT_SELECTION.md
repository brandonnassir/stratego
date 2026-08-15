# Phase 8 Agent 5 — Bounded Pilot Selection

## Role

You are **Agent 5**.

Run exactly the predeclared pilot candidate matrix and freeze one final training configuration.

This is the only Phase 8 hyperparameter-selection agent.

Do not open the test corpus for model metrics.
Do not run Phase 4 neural playing-strength evaluation.
Do not continue a winning pilot as the final Phase 8 run.

## Prerequisite

Agents 1–4 must be PASS.

Verify all candidate IDs, seeds, baseline digests, trainer version and corpus digests.

Run the full suite before your changes/harness.

## Pilot fairness

Every candidate must start from the identical canonical C1 initialization.

Every candidate must see the same ordered pilot batch identities.

Every candidate gets the same optimizer-step budget:

```text
<= 5,000 updates
```

as frozen by Agent 1.

Validation runs at exactly the same update numbers.

Do not stop a weak candidate early unless it hits a predeclared hard failure such as non-finite training.

## What may differ

Only fields already listed in Agent 1's candidate matrix.

No architecture edits.

No teacher/corpus/setup changes.

No extra "one more promising run."

## Metrics

Record each validation checkpoint:

```text
policy CE and baseline ratio
policy top-1

value CE and baseline ratio
value Brier / accuracy

belief CE and baseline ratio
belief top-1

total training loss
per-head training loss
legal policy entropy
gradient norm
learning rate
examples/s
MPS/RSS memory
```

Compute Agent 1's frozen `selection_score`.

## Selection

Apply hard veto first.

Among non-vetoed candidates:

1. lowest final validation `selection_score`;
2. tie-break lower policy ratio;
3. tie-break higher examples/s.

Do not inspect test results before selection.

Do not use game outcomes against evaluation agents as a selection signal.

## Freeze `warmstart_train_config_v1`

The winning configuration must be serialized completely:

```text
model candidate C1 + config digest
model init seed

trainer version
checkpoint version
example version
corpus version/digests

batch size
optimizer
learning rate
betas/epsilon
weight decay
gradient clip
scheduler/warmup

policy/value/belief loss weights

train shuffle seed/order version
max final updates
validation cadence
checkpoint cadence
best-checkpoint metric
early-stop rule if any

loader worker/prefetch topology
device/precision
```

The final maximum update count must be <=25,000.

Agent 6 may not tune it.

## Sanity extension

After selecting the winner, it is acceptable to run **one** additional validation-only extension of that same config, solely to determine whether the frozen final budget should be the already-predeclared shorter or longer option in Agent 1's candidate contract.

Do not create a new learning rate or loss-weight configuration.

If Agent 1 did not predeclare alternate final budgets, do not invent one.

## No test access

Add an access log/artifact proving:

```text
test examples evaluated by model in Agent 5 = 0
Phase 4 neural evaluation games Agent 5 = 0
```

Structural corpus manifests may be read.

## Artifacts

Create:

```text
reports/phase_8_data/agent_05_pilot_runs.csv
reports/phase_8_data/agent_05_pilot_selection.json
reports/phase_8_data/agent_05_frozen_train_config.json
```

Append report section 5.

## Tests

Add regression tests for:

- candidate matrix exactness;
- same initial state checksum across pilots;
- same pilot batch identity sequence;
- selection score math;
- veto logic;
- tie-break logic;
- winner is reproducible from CSV;
- train config is complete and versioned;
- test access is forbidden during selection;
- Phase 4 strength evaluation is forbidden during selection.

Run full suite after artifacts exist.

## PASS gates

PASS only if:

- candidate count <=6 and exactly matches Agent 1;
- every nonfailed candidate got equal update budget/data order;
- no unregistered config ran;
- all model-init checksums identical before training;
- selection uses validation only;
- one winner selected deterministically;
- `warmstart_train_config_v1` fully frozen;
- final update budget <=25k;
- test model inference count 0;
- Phase 4 neural strength games 0;
- full suite green.

If every candidate is vetoed or validation ratios fail to improve meaningfully, report `FAIL/BLOCKED` for review rather than broadening the search.

## Handoff to Agent 6

Provide:

```text
winning candidate id
frozen train config
all seeds
expected fresh-init checksum
final training budget
validation/checkpoint cadence
loader topology
pilot evidence
```
