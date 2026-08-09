# Phase 3 Agent 6 — Conditional Optimized Backend Prototype and Differential Validation

## Role and scope

You are **Agent 6** in a sequential Phase 3 implementation. Work only on the task in this file.

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

## CONDITIONAL TASK — do not run unless triggered

Run this agent **only** if Agent 5's data says:

```text
optimized_backend_required = true
backend_decision = BUILD_OPTIMIZED_BACKEND
```

If Agent 5 says otherwise, do not run Agent 6.

## Prerequisite

Read Agent 5's report section and:

```text
reports/phase_3_data/agent_05_end_to_end.json
```

If the required trigger is absent, mark this task `NOT REQUIRED` outside the report and stop. Do not append a fake implementation section.

## Objective

Build a **separate optimized production backend** for the measured simulator bottleneck while preserving the frozen Python engine as the behavioral source of truth.

Do not optimize the frozen reference engine in place.

## First action: profile before coding

Use Agent 5/2 timing plus targeted profiling to identify the actual bottleneck.

Record whether the dominant cost is:

- observation construction;
- behavioral processing;
- legal-action generation;
- transition logic;
- Python object overhead;
- shared-memory preparation;
- another measured component.

Do not rewrite unrelated engine modules.

## Backend location

Create a separate backend namespace, for example:

```text
stratego/backends/optimized/
```

and an adapter exposing the same production-facing operations expected by the coordinator.

The exact compiled technology may be selected based on the available toolchain and the measured hotspot, but it must:

- work natively on the target Apple Silicon machine;
- be reproducibly buildable;
- not require changing model/training semantics;
- not weaken invariant/error checks at the validation boundary.

Possible approaches include a small native extension, Cython, Numba-compatible kernels, or another justified compiled path. Choose the smallest technology that addresses the measured bottleneck.

Document the choice and build requirements.

## Required differential validation

The optimized backend must match the frozen Python reference.

Minimum acceptance:

- at least **1,000,000 randomized state comparisons**;
- zero mismatches for:
  - legal actions;
  - legal mask;
  - observation tensor;
  - behavioral metadata;
  - belief targets where exposed;
  - terminal result/reason;
- at least **10,000 complete games** with identical replay/result behavior;
- independent reset/generation semantics identical through the production adapter.

A single unexplained semantic mismatch blocks production use.

## End-to-end rebenchmark

After differential validation, rerun the best Agent 5 end-to-end benchmark configuration with the optimized backend.

Measure:

- optimized simulation-pipeline positions/second;
- optimized end-to-end positions/second;
- speedup vs Python;
- new `R`;
- MPS/worker wait fractions;
- memory;
- whether the optimization actually increases total self-play throughput.

Do not declare success based only on isolated kernel speed.

## Soak test

If the optimized backend is to be recommended for production, run the same minimum **2-hour soak** standard used by Agent 5.

## Data files

Primary:

```text
reports/phase_3_data/agent_06_optimized_backend.json
```

Optional:

```text
reports/phase_3_data/agent_06_differential_raw.csv
reports/phase_3_data/agent_06_soak_timeseries.csv
```

JSON minimum:

```text
agent
status
triggered_by_agent5
profiled_bottleneck
optimization_technology
build_requirements
state_comparisons
state_mismatches
full_games_compared
full_game_mismatches
python_simulation_positions_per_second
optimized_simulation_positions_per_second
simulation_speedup
python_end_to_end_positions_per_second
optimized_end_to_end_positions_per_second
end_to_end_speedup
R_before
R_after
soak_duration_seconds
soak_errors
memory_peak_bytes
production_backend_recommendation
files_created
files_modified
```

## Report section

Append only if this task was actually triggered and performed:

```markdown
## 6. Agent 6 — Conditional Optimized Backend
```

State explicitly whether the optimized backend is accepted for training use.

## Completion gate

PASS only if:

- optimization targets a measured bottleneck;
- frozen reference engine remains unchanged;
- differential gates have zero unexplained mismatches;
- end-to-end throughput improves enough to justify the additional backend;
- the optimized backend passes the multi-hour soak.

If the optimized backend is faster in isolation but does not materially improve end-to-end throughput, recommend keeping Python.
