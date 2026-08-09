# Phase 3 Agent 5 — End-to-End Coordinator, Soak Test, and Backend Decision

## Role and scope

You are **Agent 5** in a sequential Phase 3 implementation. Work only on the task in this file.

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

Agents 1-4 must all be `PASS`.

Read all prior Phase 3 report sections and JSON data files.

If any prerequisite is not PASS, append Section 5 as `BLOCKED` and stop.

## Objective

Integrate the completed components into the first **end-to-end bulk-synchronous self-play pipeline** and make the evidence-based decision:

> Keep the frozen Python engine for production self-play, or build a separate optimized backend?

This agent owns integration and measurement, not model training for strength.

## Files you own

Create/adapt:

```text
stratego/training/coordinator.py
stratego/training/end_to_end_benchmark.py
tests/training/test_coordinator.py
tests/training/test_end_to_end_pipeline.py
scripts/run_phase3_agent05.py
reports/phase_3_data/agent_05_end_to_end.json
```

Reuse Agents 1-4 rather than duplicating their implementations.

## Required bulk-synchronous cycle

Implement:

```text
workers build observations/legal information
-> barrier/ready state
-> coordinator runs one MPS inference batch
-> coordinator applies legality and samples actions
-> actions written to shared memory
-> workers advance assigned environments
-> completed games finalized and independently reset
-> compact decision/game records appended
-> next global step
```

Only the coordinator may own/invoke the MPS model.

Collection policy parameters remain frozen during a benchmark collection block.

Each decision record must retain `collection_policy_version`.

## Integrated correctness gate

Before performance tests, run integrated differential checks against the frozen reference/batch semantics.

At least **10,000 integrated environment steps** must have zero unexplained mismatch in:

- observation before inference;
- legal actions/mask;
- selected action legality;
- resulting state;
- terminal result/reason;
- environment generation/reset;
- trajectory identifiers.

At least **10,000 stored decisions** must reconstruct exactly through Agent 3's path.

Any correctness mismatch blocks performance conclusions.

## Configuration screening

Use Agent 2 and Agent 4 measurements to avoid wasteful full Cartesian runs.

Still evaluate enough combinations to identify interactions between CPU generation and MPS consumption.

Required dimensions:

```text
workers:      4, 6, 8, 10, 12
environments: 256, 512, 1024, 1536, 2048
inference batch sizes: 64, 128, 256, 512, 1024, 1536, 2048
```

Use:

1. short screening runs;
2. select strongest feasible configurations;
3. longer sustained measurements on finalists.

Use Agent 4's recommended precision and legality representation, but retain a dense-float32 baseline for comparison.

## Measurements

For each serious candidate record:

- end-to-end positions/second;
- games/second;
- transitions/second;
- mean/p50/p95 global-step latency;
- observation-build time;
- shared-memory/barrier time;
- MPS inference time;
- legality/action-sampling time;
- worker active fraction;
- worker barrier-wait fraction;
- coordinator/MPS active fraction;
- coordinator wait fraction;
- trajectory-recording time;
- independent resets/second;
- process memory;
- shared-memory bytes;
- MPS memory if available;
- system swap usage;
- errors/restarts.

## Define the decision ratio

Use:

\[
R =
\frac{\text{sustainable simulation-pipeline positions/second}}
{\text{sustainable representative-model inference positions/second}}.
\]

The numerator must be a **measured CPU + observation + shared-memory simulation-pipeline rate**, not the Phase 2 single-core rate multiplied by core count.

The denominator must be Agent 4's sustainable representative-model inference rate for the chosen compatible configuration.

Decision rule:

- `R >= 2.0` -> `optimized_backend_required = false`, keep Python;
- `1.25 <= R < 2.0` -> `optimized_backend_required = false`, keep Python initially and preserve the option;
- `R < 1.25` -> `optimized_backend_required = true`, run Agent 6.

Also report whether end-to-end profiling independently supports that conclusion.

## Multi-hour soak test

Run the best selected end-to-end configuration continuously for at least **2 hours**.

A longer run is optional if convenient; 2 hours is the minimum for this Phase 3 acceptance.

During the soak monitor at regular intervals:

- process memory;
- shared-memory allocation;
- MPS memory where available;
- system swap;
- throughput;
- game count;
- terminal-reason counts;
- worker liveness;
- generation/reset counts;
- coordinator/worker errors.

Acceptance:

- no invariant failure;
- no observation/state/reconstruction mismatch;
- no deadlock;
- no unexplained memory growth trend;
- no sustained swapping attributable to the pipeline;
- no unexplained material throughput collapse.

If an ordinary worker infrastructure failure occurs and recovery exists, record it. Correctness failures must stop the soak.

## Phase 3 backend decision

Your JSON/report must explicitly set one:

```text
KEEP_PYTHON
KEEP_PYTHON_OPTIMIZATION_OPTIONAL
BUILD_OPTIMIZED_BACKEND
```

Do not use qualitative wording only.

If the decision is `BUILD_OPTIMIZED_BACKEND`, recommend running Agent 6 and do not attempt Agent 6's work yourself.

## Data files

Primary:

```text
reports/phase_3_data/agent_05_end_to_end.json
```

Optional:

```text
reports/phase_3_data/agent_05_end_to_end_raw.csv
reports/phase_3_data/agent_05_soak_timeseries.csv
```

JSON minimum:

```text
agent
status
integrated_steps_checked
integrated_mismatches
stored_decisions_reconstructed
reconstruction_mismatches
screened_configurations
finalist_configurations
best_configuration
best_end_to_end_positions_per_second
best_games_per_second
simulation_pipeline_positions_per_second
representative_model_inference_positions_per_second
R
backend_decision
optimized_backend_required
worker_wait_fraction
coordinator_wait_fraction
mps_active_fraction_if_measurable
mean_step_latency_ms
p95_step_latency_ms
memory_peak_bytes
shared_memory_bytes
mps_peak_memory_bytes_if_available
swap_start_bytes
swap_end_bytes
soak_duration_seconds
soak_positions
soak_games
soak_memory_growth_bytes
soak_throughput_change_fraction
soak_errors
terminal_reason_counts
recommended_worker_count
recommended_environment_count
recommended_inference_batch_size
recommended_precision
recommended_legality_representation
recommended_snapshot_interval
files_created
files_modified
```

## Report section

Append only:

```markdown
## 5. Agent 5 — End-to-End Pipeline, Soak, and Backend Decision
```

End the section with a visible decision block:

```text
Phase 3 backend decision: <ONE OF THE THREE VALUES>
Measured R: <value>
Agent 6 required: yes/no
```

Also list any project-document updates you recommend, but **do not modify `stratego_project_docs/`**. A later review will update those documents after results are accepted.

## Completion gate

PASS only if:

- integrated correctness checks pass;
- compact trajectory reconstruction remains exact;
- scaling is measured;
- a best configuration is identified;
- the 2-hour minimum soak passes;
- `R` is measured;
- the backend decision is explicit and evidence-based.

If Agent 6 is not required, this is the normal end of Phase 3 implementation.
