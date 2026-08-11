# Phase 6 Agent 6 — Stability Soak, 168-Hour Projection, and Final Architecture Recommendation

## Role

You are **Agent 6**, the final implementation/measurement agent in sequential Phase 6.

Work from the **root of the Stratego repository**.

You may recommend Phase 6 `PASS`, `FAIL`, or `BLOCKED`, but the reviewing chat makes the final acceptance decision.

Do not begin Phase 7.

## Frozen project contracts

Do not alter:

- frozen engine/rules/observation/replay semantics;
- Agent 1 `model_contract_v2`;
- candidate architecture definitions from Agent 2;
- Phase 4 evaluation semantics;
- prior benchmark results.

Do not invent a new architecture family.

## Required reading

Read every Phase 6 artifact, especially:

```text
00_PHASE_6_SEQUENCE_AND_COMMON_CONTRACT.md
reports/phase_6_implementation_report.md
reports/phase_6_data/agent_03_inference_benchmark.csv
reports/phase_6_data/agent_03_training_step_benchmark.csv
reports/phase_6_data/agent_03_architecture_shortlist.json
reports/phase_6_data/agent_04_integrated_pipeline.csv
reports/phase_6_data/agent_04_storage_rates.csv
reports/phase_6_data/agent_04_finalists.json
reports/phase_6_data/agent_05_parallel_neural_evaluation.json
```

## Shared reporting contract

Append only:

```markdown
## 6. Agent 6 — Stability Soak, 168-Hour Projection, and Architecture Decision
```

Create:

```text
reports/phase_6_data/agent_06_soak.json
reports/phase_6_data/agent_06_soak_timeseries.csv
reports/phase_6_data/agent_06_weekly_projection.json
reports/phase_6_data/agent_06_architecture_decision.json
```

## Stop conditions

Recommend `FAIL` or `BLOCKED` instead of weakening gates if:

- any Agent 1-5 prerequisite is not PASS;
- no finalist can complete the stability soak;
- memory/swap makes every plausible candidate unsafe;
- deterministic parallel evaluation remains unresolved;
- no finalist supports a usable finite backward path;
- selecting a model would require unmeasured architectural changes.

If the leading candidate fails but a smaller existing finalist passes, selecting the smaller one is valid if fully documented.

## Prerequisite

Agents 1-5 must all be `PASS`.

Reproduce all finalist configs and parameter counts before the soak.

Run the full repository suite before Agent 6-specific scripts/changes.

## Objective

Use the accepted evidence to:

1. choose the leading finalist for a one-hour production soak;
2. verify sustained correctness/stability;
3. compare every finalist on the measured capacity/compute frontier;
4. select one exact primary model;
5. select one exact smaller fallback;
6. project the exact 168-hour final run;
7. analyze storage against the user's drives;
8. recommend Phase 6 status.

No meaningful training is authorized.

## Pre-soak candidate choice

Choose the leading soak candidate using only Agents 3-4 measured evidence:

- parameter count/capacity proxy;
- standalone inference;
- training-step throughput;
- integrated collection throughput;
- production-recording throughput;
- memory;
- storage rate;
- numerical stability.

Do not use random-weight game results.

Document why this candidate is soaked first.

## One-hour continuous soak

Run approximately one continuous hour using:

```text
real finalist model
model_contract_v2
perspective-normalized model actions
one MPS coordinator
accepted Python/shared-memory collection
production trajectory recording
trajectory_v1
snapshot interval 32
Agent 4 best defensible topology
```

Use dense legality unless Agent 4 established a clearly justified production alternative.

Do not launch a new broad tuning sweep.

## Soak measurements

Record regular time-series samples for:

```text
elapsed time
positions
positions/s
games
games/s
terminal reasons
worker failures
model/MPS failures
non-finite outputs
illegal actions
action-frame errors
reconstruction checks
process RSS
shared memory
Metal current/driver memory
swap
trajectory bytes
GiB/hour
throughput drift
```

## Hard soak targets

Required:

```text
illegal actions                         0
action-frame mismatches                 0
trajectory reconstruction mismatches    0
worker failures                         0
MPS/model failures                      0
non-finite production outputs           0
swap                                    0
unexplained persistent memory growth    0
```

Small stable/explained throughput drift is acceptable if reported.

If the leading candidate fails a candidate-specific hard gate, diagnose it and, where appropriate, soak the next finalist rather than weakening criteria.

## Finalist comparison table

For every finalist summarize:

```text
candidate ID
exact configuration
parameters
checkpoint size
standalone float32 inference
standalone float16 inference
training-step examples/s
training-step memory
integrated collection positions/s
recording-inclusive positions/s
games/s
MPS utilization
worker wait fraction
process/shared/Metal memory
GiB/hour
soak status if run
```

State explicitly that parameter count is a capacity proxy, not a proven strength measurement.

## Capacity/compute knee

Choose the architecture at the measured knee:

> the largest useful capacity increase before additional size causes disproportionate losses in self-play throughput, backward throughput, memory headroom, or stability.

For neighboring finalists quantify percentage changes in:

```text
parameters
model inference throughput
training-step throughput
recording-inclusive collection throughput
memory
storage production rate
```

Do not select merely the fastest or largest model.

Do not use random-weight effective win rate.

## Primary architecture

Select one exact production-training candidate.

Record:

```text
candidate_id
architecture_family_version
width
blocks
heads
feed_forward_width
position encoding
policy head
value head
belief head
parameter count
model_contract_v2
recommended MPS precision
recommended inference batch
recommended worker/environment topology
```

Explain the measured tradeoff.

## Fallback architecture

Select the next smaller practical architecture with:

- full correctness;
- stable MPS behavior;
- materially better throughput and/or memory headroom;
- the same model contract.

Record the same exact fields as the primary.

The fallback must be a frozen exact configuration, not an informal idea.

## 168-hour projection

The user's official final run is exactly:

\[
168\ \mathrm{hours}=604800\ \mathrm{seconds}.
\]

Using measured sustained recording-inclusive throughput `T`:

\[
N_{\mathrm{positions}}=T\times604800.
\]

Project:

```text
positions / 168 hours
games / 168 hours
trajectory GiB/hour
trajectory GiB/168 hours
checkpoint size
checkpoint storage under plausible frequencies
approximate training-step opportunities from Agent 3 measurements
```

Keep measured values and extrapolations clearly separate.

Do not promise actual learning performance from this projection.

## Storage analysis

Use the user's constraints:

```text
~150 GB free internal
~1 TB external, mostly empty
preference: preserve most games externally when practical
```

Analyze whether full measured trajectory retention fits.

If not, recommend a later retention/tiering approach such as:

```text
internal:
active checkpoint, hot logs, current buffers

external:
compressed recent full trajectories
selected older full trajectories
compact game/replay histories
evaluation and human games
diagnostic/unusual games
```

Do not invent compression ratios. Clearly distinguish measured storage from hypothetical compression examples.

Do not default to deleting most games simply because Phase 3 suggested rolling retention.

## Parallel evaluation readiness

Verify Agent 5's path is compatible with the exact primary/fallback metadata.

State whether future checkpoints are ready for deterministic:

```text
1/2/4/8-worker evaluation
greedy mode
seeded categorical mode
```

## Backend decision

Review Agent 4's candidate-specific `R` values.

State one of:

```text
KEEP_PYTHON remains supported
```

or:

```text
simulator bottleneck requires later review
```

Do not build an optimized backend here.

## Data files

`agent_06_architecture_decision.json` must include at minimum:

```text
agent
status
all_prerequisites
finalists
primary_architecture
fallback_architecture
selection_method
neighbor_tradeoffs
soak_result
weekly_projection
storage_analysis
parallel_evaluation_ready
backend_statement
full_suite
completion_gates
phase_6_recommendation
```

The soak JSON/timeseries and projection JSON must contain every headline metric used in the report.

## Final Phase 6 completion gate

Recommend Phase 6 `PASS` only if:

- Agents 1-5 all PASS;
- `model_contract_v2` is used with normalized model actions;
- candidate family was benchmarked fairly;
- standalone inference and backward evidence are complete;
- real integrated collection/recording evidence is complete;
- finalists were established without playing-strength selection;
- parallel neural evaluation is deterministic;
- selected primary completes the one-hour hard soak;
- soak illegal actions = 0;
- soak action-frame mismatches = 0;
- soak reconstruction mismatches = 0;
- soak worker/model/MPS failures = 0;
- no unexplained persistent memory growth;
- swap = 0;
- one exact primary architecture selected;
- one exact fallback architecture selected;
- decision justified from measured capacity/compute frontier;
- 168-hour positions/games/storage projection produced;
- storage constraints analyzed;
- backend status explicitly reassessed;
- full suite green;
- no meaningful training occurred;
- no playing-strength claim was made from random models.

## Final handoff to reviewing chat

Return:

```text
Agent 6 status
Phase 6 recommendation
primary exact architecture
fallback exact architecture
one-hour soak headline
168-hour compute projection
168-hour storage projection
backend statement
parallel evaluation status
full test totals
artifact paths
highest-risk remaining limitation
```

Do not modify `stratego_project_docs/`.

Do not begin Phase 7.

Only the reviewing chat may formally accept Phase 6 and freeze the architecture.
