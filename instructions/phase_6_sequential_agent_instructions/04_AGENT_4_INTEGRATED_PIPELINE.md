# Phase 6 Agent 4 — Integrated Self-Play Pipeline Benchmark

## Role

You are **Agent 4** in a sequential Phase 6 implementation for the Stratego project.

Work from the **root of the Stratego repository**.

Complete only the work assigned in this file. Do not begin later-agent tasks.

## Frozen project contracts

Do not alter:

- reference engine: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- engine action encoding: `source_destination_10000_v1`;
- Agent 1's `model_contract_v2`;
- Phase 4 match/evaluation semantics;
- Phase 3 one-MPS-owner architecture unless a blocking incompatibility is found.

Do not modify `stratego/engine/`.

## Required reading

Before implementation, read:

```text
00_PHASE_6_SEQUENCE_AND_COMMON_CONTRACT.md
reports/phase_3_implementation_report.md
reports/phase_3_data/
reports/phase_6_implementation_report.md
reports/phase_6_data/agent_01_model_contract_v2.json
reports/phase_6_data/agent_02_architecture_family.json
reports/phase_6_data/agent_03_architecture_shortlist.json
reports/phase_6_data/agent_03_inference_benchmark.csv
reports/phase_6_data/agent_03_training_step_benchmark.csv
```

Inspect the accepted Phase 3 coordinator/shared-memory/trajectory code.

## Shared reporting contract

Append only:

```markdown
## 4. Agent 4 — Integrated Self-Play Pipeline Benchmark
```

Create:

```text
reports/phase_6_data/agent_04_integrated_pipeline.csv
reports/phase_6_data/agent_04_storage_rates.csv
reports/phase_6_data/agent_04_finalists.json
```

## Stop conditions

Mark `BLOCKED` if:

- Agents 1-3 are not PASS;
- real candidates require changing frozen engine semantics;
- MPS ownership must move into simulation workers;
- normalized model actions cannot round-trip through trajectory/replay semantics;
- exact recording reconstruction cannot be maintained;
- all advancing candidates are numerically unstable in the real pipeline.

A candidate simply being slow is not a blocker.

## Prerequisite

Agents 1-3 must be `PASS`.

Reconstruct Agent 3's advancing candidates exactly.

Run the full suite and the relevant Phase 3 pipeline tests before modifications.

## Objective

Insert the real shortlisted Transformer candidates into the accepted Phase 3 bulk-synchronous pipeline and measure:

1. collection-only throughput;
2. production-recording throughput;
3. utilization/bottlenecks;
4. reconstruction correctness;
5. trajectory storage rate;
6. candidate-specific simulator/model ratio.

Then identify two or three finalists.

Do **not** choose the final primary model.

## Starting topology

Use the accepted Phase 3 starting point:

```text
backend                 KEEP_PYTHON
CPU workers             10
environments            1,536 starting point
MPS owner                coordinator only
collection              bulk synchronous
transport               persistent shared memory
live legality            dense
trajectory               trajectory_v1
snapshot interval        32
```

Do not use the old Phase 3 representative probe as a headline benchmark candidate.

## Correctness first

Before timing, prove the v2 path end to end:

```text
engine publishes absolute legal product
-> normalized model legality
-> real candidate inference
-> normalized model action selection
-> inverse perspective conversion
-> absolute engine action
-> worker applies action
```

Require:

```text
illegal selections       0
action-frame mismatches  0
model/policy errors       0
state/replay mismatches   0
```

Exercise both acting colors.

## Two benchmark modes

### Collection-only

Disable nonessential trajectory persistence to isolate sustainable simulator + inference throughput.

### Production recording

Use the actual compact future-training path:

```text
trajectory_v1
snapshot interval = 32
required sparse policy/value decision fields
```

Do not store full 127-channel observations or dense 10,000 policy vectors per decision.

Keep belief targets separate and privileged.

## Topology sweep

For each Agent 3 advancing candidate, begin with:

```text
10 workers
1,536 environments
```

Test useful environment/inference-batch points around:

```text
512
1024
1536
2048
```

where Agent 3's batch/memory results justify them.

A small worker-count sensitivity check is allowed if the real model clearly changes CPU/coordinator balance.

Do not repeat a broad CPU scaling study.

Use benchmark durations long enough to stabilize after warmup.

## Required integrated measurements

Each benchmark row should contain:

```text
candidate_id
precision
workers
environment_count
inference_batch_size
mode
duration_seconds
positions
games
positions_per_second
games_per_second
mean_game_length
terminal_reason_counts
mps_inference_fraction
host_to_device_fraction
normalized_legality_sampling_fraction
worker_active_fraction
worker_wait_fraction
trajectory_write_fraction
process_rss_bytes
shared_memory_bytes
metal_memory_bytes
swap_bytes
worker_errors
model_errors
nonfinite_outputs
illegal_actions
action_frame_errors
```

For recording rows also include:

```text
trajectory_bytes
trajectory_records
bytes_per_decision
bytes_per_game
GiB_per_hour
snapshot_count
```

## Reconstruction checks

Every headline production-recording result must sample stored games/decisions and reconstruct from accepted compact data:

```text
state
observation
absolute legal actions
normalized model legal actions
selected normalized action
selected absolute engine action
policy/value decision fields
```

Require zero mismatches.

Record exact game/decision reconstruction sample counts.

Do not store privileged belief targets in ordinary trajectory data to make this easier.

## Bottleneck ratio

For each serious candidate recompute:

\[
R=\frac{\text{sustainable simulation capacity}}{\text{sustainable candidate inference capacity}}.
\]

State exactly how numerator and denominator were measured.

Report whether:

```text
KEEP_PYTHON remains supported
```

or whether a simulator bottleneck has newly appeared and needs later review.

Do **not** build an optimized backend.

## Storage measurement

For every finalist-quality recording row compute from measured bytes/time:

```text
GiB/hour
GiB/24 hours
GiB/168 hours
```

Compare raw 168-hour production with:

```text
~150 GB free internal
~1 TB external, mostly empty
```

Do not finalize the retention policy here.

Preserve the user's preference to keep most games externally when practical.

## Finalist recommendation

Recommend two or three finalists using combined Agents 2-4 evidence:

- parameter count/capacity proxy;
- standalone inference;
- standalone backward throughput;
- integrated collection throughput;
- production-recording throughput;
- memory;
- numerical stability;
- storage production.

Do not use random-weight playing strength.

Do not choose the final primary/fallback model.

## Files you own

Suggested:

```text
stratego/training/phase6_pipeline_benchmark.py
scripts/run_phase6_agent04.py
tests/training/test_phase6_candidate_pipeline.py
reports/phase_6_data/agent_04_*.csv/json
```

Preserve one-MPS-owner design.

## Data files

`agent_04_finalists.json` should contain at minimum:

```text
agent
status
shortlist_received
correctness_gate
benchmark_topology
headline_collection_rows
headline_recording_rows
bottleneck_ratios
backend_decision_statement
reconstruction_counts
storage_projection
finalist_ids
finalist_reasons
rejected_shortlist_ids
test_total
test_passed
test_failed
files_created
files_modified
completion_gates
```

## Tests

Add regressions for:

- normalized legality/action conversion in coordinator path;
- coordinator remains sole MPS owner;
- candidate identity/config preserved;
- trajectory schema unchanged;
- v2 trajectory reconstruction;
- storage accounting;
- illegal action rejection remains loud.

Run the full suite.

## Completion gate

PASS only if:

- Agents 1-3 PASS verified;
- real advancing models used;
- v2 correctness run has 0 illegal/frame/model errors;
- collection-only benchmark completed fairly;
- production-recording benchmark completed;
- reconstruction sample has 0 mismatches;
- storage rate measured;
- candidate-specific R values computed;
- `KEEP_PYTHON` status explicitly reassessed;
- headline runs have 0 unexplained worker/model failures;
- two or three finalists identified when possible;
- no random-weight strength used;
- full suite green.

## Handoff notes for Agent 5

Provide Agent 5:

```text
one stable finalist checkpoint/config
model construction/loading API
normalized neural policy adapter
recommended inference precision
MPS ownership constraints
```

Your artifacts must also give Agent 6 the complete finalist performance/storage table.
