# Phase 3 Agent 4 — Representative PyTorch/MPS Model and Inference Benchmark

## Role and scope

You are **Agent 4** in a sequential Phase 3 implementation. Work only on the task in this file.

Assume your current working directory is the **root of the Stratego project repository**. The repository already contains the frozen Phase 2.1 reference engine and `stratego_project_docs/`.

### Frozen contracts — do not change

- Reference implementation: `phase2_1_reference_1.1.0`
- Rules: `stratego_project_v1`
- Observation: `observation_v2_1_127ch`
- Action encoding: `action_id = 100 * source + destination` over 10,000 entries
- Replay/state semantics: the frozen Phase 2.1 reference implementation

Do **not** optimize or alter `stratego/engine/` behavior. If you discover what appears to be a reference-engine correctness bug, stop and report it rather than silently fixing it.

Before implementing, read at least:

```text
stratego_project_docs/README.md
stratego_project_docs/03_game_engine_spec.md
stratego_project_docs/04_engine_validation_plan.md
stratego_project_docs/05_project_plan.md
stratego_project_docs/06_observation_v2_127ch.md
stratego_project_docs/08_internal_state_spec.md
stratego_project_docs/09_public_event_and_replay_schema.md
reports/phase_2_implementation_report.md
reports/phase_2_metrics.json
```

Also inspect the existing source and tests relevant to your task.

## Shared reporting contract

All Phase 3 agents use one Markdown report:

```text
reports/phase_3_implementation_report.md
```

Your measurements/data must go in your own files under:

```text
reports/phase_3_data/
```

Never overwrite another agent's data file.

Your report section must contain:

1. status: `PASS`, `FAIL`, or `BLOCKED`;
2. concise implementation summary;
3. files created/modified;
4. test summary;
5. key measured results;
6. deviations/limitations;
7. exact data-file paths;
8. handoff notes for the next agent.

Keep raw tables and detailed measurements in data files, not in the Markdown report.

Do not modify earlier agents' report sections except to correct an obvious formatting break that prevents the report from rendering.

## General stop conditions

Stop and mark your report section `BLOCKED` if:

- a prerequisite agent did not pass;
- frozen engine semantics would need to change;
- hidden information would need to enter a public/model-facing buffer;
- deterministic equivalence cannot be maintained;
- required platform capability is absent;
- you encounter unexplained state corruption, observation mismatch, illegal transition, or hidden-information leakage.

Do not begin the next agent's task.

## Prerequisite

Agent 3 must be `PASS`.

Read Sections 1-3 of the shared report and prior JSON outputs.

If not PASS, append Section 4 as `BLOCKED` and stop.

## Objective

Implement a **temporary representative compact Transformer** solely to benchmark Apple Metal Performance Shaders inference, legality application, and action sampling.

This is **not the final model architecture** and must be labeled accordingly in code/reporting.

Do not train it for playing strength.

## Files you own

Create/adapt:

```text
stratego/training/representative_model.py
stratego/training/mps_benchmark.py
tests/training/test_representative_model.py
scripts/run_phase3_agent04.py
reports/phase_3_data/agent_04_mps_inference.json
```

If project dependency files need PyTorch recorded, make the smallest clearly documented dependency update.

Do not change the frozen engine.

## Representative architecture

Implement approximately:

```text
100 board-square tokens
127 raw features/token
input projection -> width 128
4 Transformer encoder blocks
4 attention heads
feedforward width 512
```

Outputs:

### Policy probe

Use source-query / destination-key style scoring over 100 board locations.

Produce a logical:

```text
[B, 100, 100]
```

source-destination score matrix and flatten to 10,000 action logits.

This benchmark head should exercise the same general action-space cost expected later.

### Value probe

Produce three logits/probabilities for:

```text
win
draw
loss
```

### Belief probe

Use a lightweight shared-head placeholder that produces per-square opponent-type logits sufficient to exercise representative shared-belief compute.

Do not build the paper's separate large belief Transformer.

## Correctness tests

Verify on CPU and MPS where available:

- output shapes;
- finite outputs;
- dense legality correctly eliminates illegal actions;
- sampled action is always legal;
- value output has three classes;
- same deterministic input/model state gives numerically stable repeated output for the same precision/device;
- no hidden engine data enters the model beyond the approved observation/legal inputs.

This is not a bitwise CPU-vs-MPS equivalence requirement.

## MPS requirement

This agent is expected to run on the target Apple M4 Pro.

Detect MPS availability rather than assuming it.

If MPS is unavailable, mark `BLOCKED` and stop; do not substitute CPU results for the required benchmark.

Synchronize the MPS device around timed regions so asynchronous execution does not produce false timings.

## Batch-size benchmark

Benchmark inference batches:

```text
64
128
256
512
1024
1536
2048
```

For each feasible batch measure:

- warm-up;
- synchronized latency;
- positions/second;
- model-only positions/second;
- legality + sampling positions/second;
- peak process memory;
- MPS/device memory if measurable;
- failures/out-of-memory conditions.

Use enough iterations to avoid one-shot timing noise without making the benchmark unnecessarily long.

## Precision benchmark

Baseline:

```text
float32
```

Also test supported reduced precision, especially:

```text
float16
```

only if the actual model path and required operations are supported and stable on the target device.

Do not claim a reduced-precision win merely from one fast operation. Compare the complete representative inference + legality + action-sampling path.

Record unsupported or unstable modes explicitly.

## Dense vs compact legality benchmark

Dense baseline:

```text
[B, 10000] legality mask
```

Then benchmark a compact legality path using legal action identifiers in a padded or otherwise vectorized representation.

The compact path must produce legal samples and equivalent normalized probabilities over the legal set.

Compare:

- latency;
- positions/second;
- memory;
- complexity/robustness.

Do not force sparse legality into later agents if it provides no meaningful end-to-end benefit.

## Data files

Primary:

```text
reports/phase_3_data/agent_04_mps_inference.json
```

Optional raw:

```text
reports/phase_3_data/agent_04_mps_inference_raw.csv
```

JSON minimum:

```text
agent
status
platform
torch_version
mps_available
mps_device_info_if_available
representative_model_parameter_count
architecture_summary
batch_sizes
precision_modes
dense_legality_results
compact_legality_results
best_dense_configuration
best_compact_configuration
recommended_legality_representation
recommended_precision
best_inference_positions_per_second
best_end_to_end_model_step_positions_per_second
peak_memory_bytes
mps_peak_memory_bytes_if_available
failures
files_created
files_modified
```

## Report section

Append only:

```markdown
## 4. Agent 4 — Representative MPS Inference Benchmark
```

Clearly state that the network is a benchmark probe, not a frozen model design.

## Completion gate

PASS only if:

- MPS is available;
- representative model outputs/legality/sampling are correct;
- all feasible required batch sizes are benchmarked;
- float32 baseline is measured;
- reduced precision is tested only where actually supported;
- dense vs compact legality is compared;
- a best sustainable representative-model inference rate is recorded.

Do not build the end-to-end multiprocess coordinator.
