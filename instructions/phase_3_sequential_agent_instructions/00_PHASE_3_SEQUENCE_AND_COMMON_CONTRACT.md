# Phase 3 Sequential Agent Plan

## Purpose

Phase 3 determines whether the frozen Python reference engine can support high-throughput self-play on the target Apple M4 Pro Mac mini, or whether a second optimized simulator backend is actually necessary.

Use the agents **sequentially**, never concurrently.

## Frozen project state

The following are frozen before Phase 3:

- `phase2_1_reference_1.1.0`
- `stratego_project_v1`
- `observation_v2_1_127ch`
- 10,000-entry source-destination action encoding
- Phase 2.1 replay/state semantics

The Python engine is a **correctness oracle**. Phase 3 wraps and benchmarks it; it does not optimize it in place.

## Approved architecture

Phase 3 implements these four decisions:

1. **Bulk-synchronous collection**
2. **One Metal-owning coordinator process plus multiple CPU simulation workers**
3. **Persistent preallocated shared-memory buffers**
4. **Compact trajectories with periodic snapshots rather than stored full observations**

Initial engineering point:

- 1,024 simultaneous environments
- approximately 8 CPU workers
- dense 10,000-entry legality masks as the first live-transport baseline
- snapshot every 32 plies

These are starting values, not final values.

## Agent sequence

| Agent | Task | Run? |
|---|---|---|
| 1 | Batch wrapper, independent resets, reference equivalence | Always |
| 2 | Multiprocess workers, shared memory, CPU-side scaling | After Agent 1 PASS |
| 3 | Compact trajectory/snapshot/reconstruction system | After Agent 2 PASS |
| 4 | Representative PyTorch/MPS model and inference benchmarks | After Agent 3 PASS |
| 5 | End-to-end coordinator, scaling, soak, backend decision | After Agent 4 PASS |
| 6 | Separate optimized backend prototype + differential validation | **Only if Agent 5 says required** |

If Agent 5 measures:

\[
R =
rac{	ext{sustainable simulation-pipeline positions/second}}
{	ext{sustainable representative-model inference positions/second}},
\]

use this decision rule:

- `R >= 2.0`: keep Python as production simulator;
- `1.25 <= R < 2.0`: keep Python initially; optimized backend remains optional;
- `R < 1.25`: Agent 6 is required.

Do **not** run Agent 6 merely because compiled code might be faster.

## Shared report

All agents append to:

```text
reports/phase_3_implementation_report.md
```

Required section ownership:

```text
# Phase 3 Implementation Report

## 1. Agent 1 — Batch Wrapper and Reference Equivalence
## 2. Agent 2 — Shared Memory and CPU Scaling
## 3. Agent 3 — Trajectory Storage and Reconstruction
## 4. Agent 4 — Representative MPS Inference Benchmark
## 5. Agent 5 — End-to-End Pipeline, Soak, and Backend Decision
## 6. Agent 6 — Conditional Optimized Backend
```

Agent 1 creates the report header. Later agents append only their section.

## Separate data files

Use these canonical files:

```text
reports/phase_3_data/agent_01_batch_equivalence.json
reports/phase_3_data/agent_02_shared_memory_scaling.json
reports/phase_3_data/agent_03_trajectory_reconstruction.json
reports/phase_3_data/agent_04_mps_inference.json
reports/phase_3_data/agent_05_end_to_end.json
reports/phase_3_data/agent_06_optimized_backend.json
```

Agents may add uniquely named raw CSV/JSON files in the same directory, but must never reuse another agent's filename.

## Global Phase 3 acceptance goals

By the end of Agent 5 (or Agent 6 if triggered), we need measured answers to:

- Does batching preserve exact reference behavior?
- Do finished environments reset independently?
- Can CPU workers exchange observations/actions through persistent shared memory without serialization becoming the bottleneck?
- Can compact trajectories reconstruct exact historical observations/legal actions/belief targets?
- What worker count is fastest?
- What environment count is fastest?
- What inference batch size is fastest?
- What precision is useful on the actual MPS device?
- Is dense or compact legality faster end to end?
- What is sustainable end-to-end positions/second?
- What fraction of time do workers/coordinator wait?
- Is there unexplained memory growth during a multi-hour soak?
- What is the measured ratio `R`?
- Do we keep Python or build a second optimized backend?

Do not modify the frozen observation/rules/action contracts to improve benchmark results.
