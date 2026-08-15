# Phase 8 Agent 4 — MPS Trainer, Checkpoint/Resume, and Throughput Validation

## Role

You are **Agent 4**.

Build the actual C1 warm-start trainer and prove it can update, checkpoint, resume, and run stably on the M4 Pro.

Do not perform pilot selection.
Do not use the test corpus.
Do not use Phase 4 neural playing strength.

## Prerequisite

Agents 1–3 must be PASS.

Verify corpus/example/baseline digests.

Run the full suite before edits.

## Mission

Implement:

```text
warmstart_trainer_v1
warmstart_checkpoint_v1
```

using:

```text
C1
model_contract_v2
MPS
float32
AdamW
batch size 256 reference
```

The trainer must support the Agent 1 predeclared pilot candidates without architecture changes.

## Loss implementation

Implement exactly the Agent 1/3 target semantics:

```text
masked legal policy CE
categorical WDL CE
hidden-only belief CE
```

Normalize each loss independently before loss weighting.

Report per batch:

```text
policy loss
value loss
belief loss
total loss

policy supervised decisions
value decisions
belief hidden-piece targets

legal policy entropy
gradient norm
learning rate
parameter norm
```

Never average belief over all 100 board squares with ignored squares silently contributing zeros to the denominator.

## Policy legality masking

The policy CE denominator must include only legal model-frame actions.

Tests must prove:

```text
adding arbitrarily large logits on illegal actions
does not change policy loss
```

and:

```text
teacher action must be legal
```

Fail loudly otherwise.

## Optimizer / scheduler

Support only Agent 1's predeclared candidate family.

No ad hoc config added after seeing pilot results.

Use explicit serializable config.

Gradient clipping is required and its pre/post values should be logged.

No PPO, KL, advantage, EMA/self-play damping, or setup loss.

## Checkpoint contents

Implement `warmstart_checkpoint_v1` with every state listed in the common contract.

Checkpoint load must reject mismatches in:

```text
C1 config digest
model contract
observation version
corpus version/digest
example version
trainer version
train config identity
```

unless the caller invokes an explicit evaluation-only load path for a compatible model checkpoint.

Do not silently resume on a different corpus.

## Resume equivalence

Run a deterministic split-run experiment.

Recommended:

```text
uninterrupted: 1,000 updates

split:
    400 updates
    save
    destroy trainer process/state
    reload
    600 updates
```

Require:

```text
same batch identities at every compared step
same global step
same examples consumed
same LR/scheduler state
same validation cadence
same best-checkpoint logic
same optimizer state structure
```

Parameter equivalence:

```text
torch.allclose(
    resumed,
    uninterrupted,
    rtol=1e-5,
    atol=1e-6
)
```

for every parameter, with recorded max absolute/relative differences.

If the MPS path is exactly equal, report that stronger result.

Also run a smaller CPU exact-determinism reference if practical.

## Interrupted checkpoint writes

Checkpoint writing must be atomic or equivalent:

```text
write temporary file
flush/sync as appropriate
validate
rename/commit
```

A crash during checkpoint write must leave either the last complete checkpoint or the new complete checkpoint—never a file accepted as valid but partially written.

Add corrupted/truncated checkpoint rejection tests.

## Trainer throughput benchmark

Use real reconstructed train examples.

Benchmark at least:

```text
C1 float32
batch 256
```

Optionally benchmark one larger batch only if Agent 1's candidate contract permits it.

Measure after warmup:

```text
forward ms
loss ms
backward ms
optimizer ms
data wait ms
validation overhead
examples/s
updates/s
peak RSS
MPS allocated/driver memory if available
```

Do not compare architecture candidates.

## Loader/trainer balance

Measure whether the MPS waits on data.

If data wait exceeds 15% of training wall time, tune worker/prefetch settings within the frozen dataset semantics.

Do not change selected-example identity/order.

If still data-bound, report the bottleneck for review.

## Numerical stability soak

Run at least:

```text
2,000 optimizer updates
```

using one neutral Agent-1 candidate configuration on train data.

This is not candidate selection.

Require:

```text
non-finite losses       0
non-finite gradients    0
non-finite parameters   0
illegal targets         0
data mismatches         0
checkpoint errors       0
```

Record loss trends descriptively but do not choose the Phase 8 config from them.

## Validation implementation

Implement validation metrics against Agent 3's frozen baselines.

Validation must:

```text
run no_grad
not update optimizer/scheduler
not change train data cursor
not change model mode permanently
aggregate by game for bootstrap support
```

Do not evaluate test.

## Suggested files

```text
stratego/training/warmstart_loss.py
stratego/training/warmstart_trainer.py
stratego/training/warmstart_checkpoint.py
stratego/training/warmstart_metrics.py

tests/training/test_warmstart_loss.py
tests/training/test_warmstart_trainer.py
tests/training/test_warmstart_checkpoint.py

scripts/run_phase8_agent04.py
```

## Artifacts

Create:

```text
reports/phase_8_data/agent_04_trainer_contract.json
reports/phase_8_data/agent_04_training_benchmark.csv
reports/phase_8_data/agent_04_resume_validation.json
```

Append report section 4.

## PASS gates

PASS only if:

- Agents 1–3 PASS;
- finite C1 MPS optimizer path;
- three losses match frozen semantics;
- illegal logits cannot affect policy loss;
- loss normalization exact;
- trainer supports only predeclared candidate matrix;
- checkpoint includes complete logical state;
- mismatch/corruption rejection works;
- interrupted checkpoint write safe;
- 1,000-step split-run resume comparison passes;
- exact next-batch/data-cursor resume passes;
- >=2,000-update MPS stability soak has zero non-finite failures;
- throughput/memory/data-wait measured;
- validation cannot mutate training state;
- test corpus untouched by model inference;
- full suite green.

## Handoff to Agent 5

Provide:

```text
trainer API
checkpoint API/version
candidate-config API
throughput
recommended fixed loader topology
resume evidence
validation API
all candidate IDs frozen by Agent 1
```
