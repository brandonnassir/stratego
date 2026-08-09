# Phase 3 Agent 2 — Shared Memory, Multiprocessing, and CPU Scaling

## Role and scope

You are **Agent 2** in a sequential Phase 3 implementation. Work only on the task in this file.

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

Read:

```text
reports/phase_3_implementation_report.md
reports/phase_3_data/agent_01_batch_equivalence.json
```

Agent 1 must be `PASS`.

If not, append your Section 2 as `BLOCKED` and stop.

## Objective

Turn Agent 1's batch semantics into a **multiprocess CPU simulation layer** using persistent preallocated shared-memory buffers.

This agent does **not** implement the neural network or MPS inference.

## Files you own

Create/adapt:

```text
stratego/training/shared_buffers.py
stratego/training/worker_pool.py
tests/training/test_shared_buffers.py
tests/training/test_worker_pool.py
scripts/run_phase3_agent02.py
reports/phase_3_data/agent_02_shared_memory_scaling.json
```

Do not change Agent 1's public batch semantics unless a bug is proven. If you need an interface extension, make it backward-compatible and document it.

## Architecture requirement

Use:

```text
coordinator process
    |
persistent shared-memory arrays
    |
CPU simulation workers
```

Workers own disjoint environment ranges.

The shared-memory payload must include enough information for later model inference, initially:

```text
observations        [N, 127, 10, 10]  float32
dense_legal_mask    [N, 10000]        bool/uint8
acting_player       [N]               compact integer
environment_id      [N]
generation          [N]
actions             [N]               integer
terminal/status metadata as needed
```

Exact dtypes may be adjusted for correctness/efficiency but must be documented.

### No object-payload queues

Do not send full observations, legal masks, or game states through multiprocessing queues/pipes on each step.

Small control messages/primitives are permitted for:

- worker startup/shutdown;
- errors;
- barrier/phase signaling;
- benchmark commands.

Bulk data must remain in persistent shared memory.

## Worker behavior

Each worker:

1. owns a fixed set of environment slots;
2. reads actions for those slots;
3. advances games using the frozen reference engine/batch semantics;
4. independently resets finished slots when instructed by the coordinator policy;
5. writes the next observation/legal metadata into its fixed shared-memory slice;
6. signals completion.

Set worker-side thread counts to avoid accidental CPU oversubscription if any numerical libraries spawn their own threads.

All randomness must be deterministically seeded from explicit root seed + worker/environment/generation information.

## Correctness tests

At minimum:

### Shared buffer round trip

Write known arrays in workers and verify coordinator sees exact contents without serialization transforms.

### Multiprocess equivalence

For at least **25,000 cross-process environment steps**, compare shared-memory results with Agent 1's batch wrapper/reference behavior.

Require zero mismatch for:

- observations;
- legal masks/lists where checked;
- acting player;
- environment generation;
- terminal result/reason;
- selected next transition.

### Reset isolation

At least **5,000 reset events** distributed across workers.

Require:

- only intended slot resets;
- generation increments once;
- neighboring slots unchanged;
- no deadlock.

### Worker failure surface

Deliberately terminate one test worker.

The coordinator must detect it and return a clear infrastructure error. Do not silently continue with stale buffers.

Do not implement full production recovery yet.

## CPU scaling benchmark

Benchmark the CPU/shared-memory path **without MPS**.

Screen:

```text
workers:      4, 6, 8, 10, 12
environments: 256, 512, 1024, 1536, 2048
```

You do not need a long run for every Cartesian pair.

Recommended approach:

1. short screening of all feasible pairs;
2. longer measurement of the best few;
3. record both per-step latency and sustainable positions/second.

Use a cheap deterministic legal-action selection policy for the benchmark so model inference is not involved.

Measure:

- positions/second;
- state transitions/second;
- observation builds/second;
- coordinator wait fraction;
- worker active fraction if practical;
- barrier wait fraction;
- process CPU utilization;
- process/shared-memory memory use;
- errors/deadlocks.

Do not infer scaling by multiplying the single-process Phase 2 rate.

## Data files

Primary:

```text
reports/phase_3_data/agent_02_shared_memory_scaling.json
```

If useful, add:

```text
reports/phase_3_data/agent_02_shared_memory_scaling_raw.csv
```

JSON must include at minimum:

```text
agent
status
platform
python_version
worker_counts
environment_counts
shared_buffer_shapes
shared_buffer_dtypes
cross_process_steps
equivalence_mismatches
reset_events
reset_mismatches
worker_failure_detection_passed
screening_results
best_cpu_configuration
best_cpu_positions_per_second
best_cpu_transitions_per_second
memory_peak_bytes
deadlocks
errors
files_created
files_modified
```

## Report section

Append only:

```markdown
## 2. Agent 2 — Shared Memory and CPU Scaling
```

Keep detailed matrix data in JSON/CSV.

## Completion gate

PASS only if:

- Agent 1 semantics are preserved;
- at least 25,000 cross-process steps show zero unexplained mismatch;
- at least 5,000 independent resets pass;
- persistent shared-memory transport works;
- worker failure is detectable;
- worker/environment scaling is measured;
- no deadlock or stale-buffer correctness issue remains.

Do not implement PyTorch/MPS inference.
