# Phase 8 Agent 6 — Canonical C1 Warm-Start Run

## Role

You are **Agent 6**.

Run the one canonical Phase 8 warm-start training job from a fresh C1 random initialization using the exact Agent 5 frozen configuration.

Freeze the best validation checkpoint.

Do not use test metrics.
Do not use the Phase 4 evaluation bank to select the checkpoint.
Do not begin self-play.

## Prerequisite

Agents 1–5 must be PASS.

Verify:

```text
warmstart_train_config_v1
corpus digests
example version
trainer/checkpoint versions
canonical C1 init checksum
C1 config digest
```

Run the full suite before the run.

## Fresh-start requirement

Reconstruct C1 from the Agent 1 canonical initialization seed.

Require the pre-training state checksum to equal Agent 5's recorded expected checksum.

Do not load pilot model weights.

Optimizer/scheduler start fresh.

## Training data

Use only:

```text
synthetic_warmstart_corpus_v1 / train
```

Weight updates must never consume validation/test examples.

Use Agent 3's deterministic selected-example universe/order.

## Validation / checkpoint selection

At the frozen cadence:

```text
evaluate validation corpus
compute frozen baseline ratios
compute frozen selection score
write checkpoint
```

The final accepted Phase 8 checkpoint is:

```text
the checkpoint with the best frozen validation selection score
```

subject to finite/correct training.

Do not select by random EWR.

Do not inspect test.

## Training logging

For every reporting interval record:

```text
wall time
optimizer step
examples consumed

policy/value/belief/total train losses
validation losses + ratios

policy top-1
value accuracy/Brier
belief top-1
legal policy entropy

learning rate
gradient norm pre/post clip
parameter norm

examples/s
data wait fraction
RSS
MPS memory
```

Record terminal/teacher/setup distributions of consumed data only as corpus statistics; the corpus itself is static.

## Checkpoint cadence and resume proof

During the canonical run, perform at least one real stop/restart exercise if the frozen run budget is long enough:

```text
save normal checkpoint
terminate trainer process cleanly
reload
continue
```

The resumed run must preserve logical cursor/counters.

This is not a new Agent 4 equivalence experiment; it confirms the production path uses the accepted resume machinery.

If the whole final budget finishes too quickly for a meaningful intentional restart, a separate same-config production resume rehearsal is acceptable.

## Failure behavior

Stop the run and report if:

```text
non-finite loss
non-finite gradient
non-finite parameter
target mismatch
checkpoint corruption
data cursor inconsistency
validation leak
MPS failure
```

Do not skip a batch and silently continue.

## Canonical checkpoint freeze

At completion:

1. identify best validation checkpoint;
2. reload it independently;
3. verify architecture/config/dataset identities;
4. run validation again and require reproduced metrics;
5. compute file SHA-256;
6. write immutable checkpoint manifest.

Preferred:

```text
checkpoints/phase8/warmstart_c1_v1.pt
checkpoints/phase8/warmstart_c1_v1_manifest.json
```

Use repository conventions if different.

## What you may report

You may report:

```text
train metrics
validation metrics
training stability
checkpoint identity
```

You may **not** report:

```text
test model metrics
Phase 4 random/basic/tactical/strategic neural strength
```

Those belong to Agent 7.

## Artifacts

Create:

```text
reports/phase_8_data/agent_06_warmstart_run.json
reports/phase_8_data/agent_06_training_curve.csv
reports/phase_8_data/agent_06_checkpoint_manifest.json
```

Append report section 6.

## PASS gates

PASS only if:

- Agents 1–5 PASS;
- final run starts from exact fresh C1 init;
- exact frozen config used;
- train split only updates weights;
- validation only selects checkpoint;
- no test model inference;
- no Phase 4 neural strength evaluation;
- zero non-finite/target/checkpoint/data errors;
- real checkpoint/restart path exercised successfully;
- best checkpoint reload reproduces validation metrics;
- checkpoint digest/manifest written;
- checkpoint differs materially from initialization;
- no Phase 9 self-play/RL machinery used;
- full suite green.

## Handoff to Agent 7

Provide:

```text
frozen checkpoint path
checkpoint SHA-256
checkpoint manifest
initial checkpoint identity
frozen train config
training curve
best validation metrics
proof test/Phase4 strength remained sealed
```
