# Phase 6 Agent 3 — Standalone MPS Inference and Training-Step Benchmark

## Role

You are **Agent 3** in a sequential Phase 6 implementation for the Stratego project.

Work from the **root of the Stratego repository**.

Complete only the work assigned in this file. Do not begin later-agent tasks.

## Frozen project contracts

Do not alter Agents 1-2 accepted model semantics or candidate configurations.

Do not modify `stratego/engine/` or Phase 4 evaluation semantics.

## Required reading

Read:

```text
00_PHASE_6_SEQUENCE_AND_COMMON_CONTRACT.md
reports/phase_6_implementation_report.md
reports/phase_6_data/agent_01_model_contract_v2.json
reports/phase_6_data/agent_02_architecture_family.json
reports/phase_5_implementation_report.md
```

Inspect Agent 2's model/config code and Phase 5 MPS benchmark methodology.

## Shared reporting contract

Append only:

```markdown
## 3. Agent 3 — Standalone MPS and Training-Step Benchmark
```

Create:

```text
reports/phase_6_data/agent_03_inference_benchmark.csv
reports/phase_6_data/agent_03_training_step_benchmark.csv
reports/phase_6_data/agent_03_architecture_shortlist.json
```

## Stop conditions

Mark `BLOCKED` if:

- Agents 1 or 2 are not PASS;
- candidate configs/parameter counts do not reproduce;
- MPS is unavailable on the target M4 Pro;
- benchmark rows cannot be made comparable/reproducible;
- all candidates fail numerically on MPS.

A large candidate being too slow or out-of-memory is a valid result, not a blocker.

## Prerequisite

Agents 1-2 must be `PASS`.

Reconstruct every candidate from Agent 2's stored config and require exact config digest and parameter count before timing.

Run the full suite before benchmark-code changes.

## Objective

Measure the M4 Pro hardware frontier of C0-C6 **without the simulator** and produce a measurement-based shortlist for integrated testing.

Do not select the final architecture.

## Benchmark fairness

Use identical:

```text
input corpus
initialization policy
warmup policy
timing boundaries
MPS synchronization
measurement repetitions
dtype conventions
tokenization/legal-action path
loss definitions
```

across candidates.

Record exact hardware/software versions and memory APIs.

Do not optimize a favored candidate differently.

## Input corpus

Create a deterministic corpus of valid engine-derived observations and legal masks from real positions.

The same rows must be used across candidate and precision comparisons.

Record a corpus seed/digest.

Privileged belief targets may be generated only for the backward benchmark target path.

## Inference matrix

Attempt each viable candidate at:

```text
batch sizes: 1, 64, 256, 512, 1024, 1536, 2048
precision:   float32, float16
```

Continue upward until out-of-memory, documented practical memory limit, or repeated correctness failure.

OOM/error rows must remain in the CSV.

## Timing boundaries

Where practical report three boundaries:

### A. Model forward only

Input already prepared/device-ready.

### B. Observation + tokenization + model

Includes required model-facing preprocessing.

### C. Observation + tokenization + model + normalized legality/action selection

Closer to live inference, still without simulation.

Record exactly what each benchmark row includes.

Synchronize MPS around timed regions so asynchronous execution does not corrupt latency.

Warm up before measurements.

## Inference measurements

Each row should contain:

```text
candidate_id
architecture/config digest
precision
batch
boundary
status
warmup_iterations
measurement_iterations
median_latency_ms
p95_latency_ms
mean_latency_ms
positions_per_second
finite_outputs
process_rss_bytes
metal_allocated_bytes
metal_driver_bytes
peak_memory_if_available
oom
error
```

Unavailable memory metrics must be marked unavailable, not zero.

## Numerical checks

For every serious candidate compare:

```text
CPU float32 vs MPS float32
float32 reference vs MPS float16
```

Predeclare tolerances.

Report per head:

```text
max absolute error
mean absolute error
meaningful relative error
finite-output status
```

Also report:

- crafted-margin greedy action agreement;
- natural-corpus greedy agreement;
- legal absolute action validity after normalized selection/conversion.

Do not fail a valid float16 run solely because near-zero logits create large relative ratios; report absolute error honestly.

Do not ignore crafted-margin action flips.

## Training-step benchmark

For every serious candidate attempt:

```text
batch sizes: 32, 64, 128, 256
precision:   float32 and float16 where backward is supported
```

One step contains:

```text
forward
policy loss
W/D/L value loss
masked hidden-only belief loss
backward
```

No optimizer step.

No parameter update.

## Training measurements

Record:

```text
candidate_id
precision
batch
status
forward_ms
loss_ms
backward_ms
total_ms
examples_per_second
policy_loss
value_loss
belief_loss
total_loss
finite_loss
finite_gradients
shared_encoder_gradient
policy_head_gradient
value_head_gradient
belief_head_gradient
process_memory
metal_memory
oom
error
```

Never silently run float32 while labeling float16.

## Candidate classification

Classify every candidate:

```text
ADVANCE
DOMINATED
IMPRACTICAL
```

Use a documented deterministic rule based on:

- parameter count as a capacity proxy;
- sustainable inference throughput;
- training-step throughput;
- memory;
- numerical stability;
- usable batch range.

Do not use random-weight game performance.

Aim to advance at least three genuinely viable candidates. If fewer are viable, advance all viable candidates and explain.

Measure C6 far enough to establish why it is or is not practical.

## Pareto/frontier summary

For each candidate report:

```text
parameters
best stable float32 inference positions/s
best stable float16 inference positions/s
representative training examples/s
max stable inference batch
max stable training batch
memory
classification
reason
```

Identify the frontier without naming the final production model.

## Files you own

Suggested:

```text
stratego/model/benchmark_helpers.py
scripts/run_phase6_agent03.py
tests/model/test_phase6_benchmarks.py
reports/phase_6_data/agent_03_*.csv/json
```

Do not edit candidate architectures to improve their benchmark results.

## Data files

The shortlist JSON must include at minimum:

```text
agent
status
candidate_configs
input_corpus_digest
benchmark_method
numerical_tolerances
candidate_summaries
pareto_frontier
advance_ids
dominated_ids
impractical_ids
classification_rules
ooms
numerical_failures
recommended_integrated_test_configs
test_total
test_passed
test_failed
files_created
files_modified
completion_gates
```

## Tests

Add tests proving:

- benchmark configs reproduce Agent 2 configs;
- benchmark input corpus is deterministic;
- MPS timing synchronizes;
- CPU cannot be mislabeled as MPS;
- precision labels reflect actual dtype/path;
- OOM rows are retained;
- classification is deterministic from stored summaries;
- no strength field influences classification.

Run the full suite.

## Completion gate

PASS only if:

- Agents 1-2 PASS verified;
- candidate configs/parameter counts reproduce;
- MPS actually used;
- deterministic valid corpus recorded;
- fair inference matrix attempted;
- OOM/error rows retained;
- CPU/MPS numerical checks completed;
- float16 honestly tested;
- crafted-margin action agreement passes for advancing candidates;
- training-step benchmark completed for serious candidates;
- losses/gradients finite for advancing candidates;
- memory measured/reported;
- deterministic classification produced;
- at least three advance when at least three are viable;
- no strength-based selection occurred;
- full suite green.

## Handoff notes for Agent 4

Provide:

```text
ADVANCE candidate IDs/configs
best stable precision
usable inference batch ranges
parameter counts
training-step throughput
memory limits
numerical caveats
recommended integrated benchmark starting configs
```

Agent 4 must not resurrect dominated candidates without a specific measured integration reason.
