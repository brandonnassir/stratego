# Phase 6 Agent 2 — Candidate Architecture Family

## Role

You are **Agent 2** in a sequential Phase 6 implementation for the Stratego project.

Work from the **root of the Stratego repository**.

Complete only the work assigned in this file. Do not begin later-agent tasks.

## Frozen project contracts

Do not alter:

- reference engine: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- engine action encoding: `source_destination_10000_v1`;
- Phase 4 evaluation semantics;
- Agent 1's accepted `model_contract_v2` action-frame conversion.

Do not modify `stratego/engine/`.

## Phase 6 scope

Your task is to build the **configurable candidate model family** that Agent 3 will benchmark.

You do not select the production model.

You do not perform broad performance benchmarking or meaningful training.

## Required reading

Before implementation, read:

```text
00_PHASE_6_SEQUENCE_AND_COMMON_CONTRACT.md
reports/phase_6_implementation_report.md
reports/phase_6_data/agent_01_model_contract_v2.json
reports/phase_5_implementation_report.md
stratego/model/
```

Also inspect existing Phase 5 model/checkpoint tests.

## Shared reporting contract

Append only:

```markdown
## 2. Agent 2 — Candidate Architecture Family
```

to:

```text
reports/phase_6_implementation_report.md
```

Write:

```text
reports/phase_6_data/agent_02_architecture_family.json
```

## Stop conditions

Mark `BLOCKED` if:

- Agent 1 is not `PASS`;
- Agent 1's v2 public APIs are insufficient without changing the engine;
- the common candidate family requires privileged inputs;
- model semantics would have to differ across candidates;
- even small candidates cannot construct/run on target MPS for reasons not attributable to their size.

An individual large candidate failing due to size is not a blocker.

## Prerequisite

Agent 1 must be `PASS`.

Read:

```text
reports/phase_6_data/agent_01_model_contract_v2.json
```

Confirm:

```text
model contract       model_contract_v2
token frame          perspective-normalized
policy frame         perspective-normalized
engine frame         absolute
```

Run the full suite before edits and record totals.

## Objective

Create one configurable Transformer family with candidates:

| ID | Width | Blocks | Heads | Feed-forward width | Role |
|---|---:|---:|---:|---:|---|
| C0 | 64 | 2 | 4 | 256 | small control |
| C1 | 128 | 4 | 4 | 512 | small practical |
| C2 | 192 | 4 | 6 | 768 | wider |
| C3 | 192 | 6 | 6 | 768 | deeper |
| C4 | 256 | 6 | 8 | 1024 | medium-large |
| C5 | 256 | 8 | 8 | 1024 | deeper medium-large |
| C6 | 384 | 8 | 8 | 1536 | paper-width/depth ceiling reference |

C6 is an upper-region benchmark reference, not a presumed choice.

If a literal row violates a hard PyTorch constraint, make the smallest necessary adjustment, record it prominently, and keep the intended scaling ladder.

## One implementation, not seven forks

Create one serializable configuration source of truth with at minimum:

```text
candidate_id
width
blocks
heads
feed_forward_width
input_channels
board_tokens
policy_size
value_classes
belief_classes
position_encoding
normalization
dropout
architecture_family_version
```

Candidate construction must be deterministic from:

```text
configuration + explicit seed
```

Do not maintain seven hand-edited model classes.

## Fixed high-level architecture

All candidates share:

```text
[B,127,10,10]
-> [B,100,127] normalized tokens
-> input projection to width D
-> learned row embedding + learned column embedding
-> pre-normalization Transformer encoder
-> shared 100-token board representation
   ├─ policy [B,10000]
   ├─ value  [B,3]
   └─ belief [B,100,12]
```

No causal mask.

No candidate-specific extra observation features.

No absolute Red/Blue input feature.

## Position representation

Use learned row and column embeddings:

\[
h_{r,c}=W_xx_{r,c}+e_r+e_c.
\]

Pin normalized row/column indexing with tests.

Do not use an absolute-color orientation that undoes Agent 1's symmetry decision.

## Transformer blocks

Use a fixed standard pre-normalization block design:

```text
LayerNorm
-> multi-head self-attention
-> residual
-> LayerNorm
-> feed-forward
-> residual
```

Keep activation, normalization epsilon, biases, dropout policy, and attention implementation identical across candidates.

If dropout exists, deterministic benchmark/evaluation mode must disable randomness. Zero dropout is acceptable.

## Policy head

Use a source-query / destination-key head shared across candidates.

For encoded tokens `H`:

\[
Q=HW_Q,
\qquad
K=HW_K
\]

and a source/destination logit matrix such as:

\[
L_{ij}=\frac{Q_i^\top K_j}{\sqrt{d_q}}+b_i^{(s)}+b_j^{(d)}.
\]

Flatten `[100,100]` to `[10000]` in model-frame source/destination order.

Do not insert absolute engine conversion inside the network.

## Value head

Use the same design for all candidates, for example:

```text
mean pool 100 encoded tokens
-> small MLP
-> [WIN, DRAW, LOSS]
```

The categorical value contract remains unchanged.

## Belief head

Use a lightweight per-token head:

```text
encoded token -> 12 logits
```

Output:

```text
[B,100,12]
```

Keep Phase 5 hidden-only belief supervision semantics.

Do not build the paper's separate large belief decoder during Phase 6.

## Architecture identity and checkpoint compatibility

Create an architecture-family identifier distinct from:

```text
integration_model_v1
```

Each candidate must carry enough config identity that incompatible candidates fail checkpoint loading even if some tensor shapes happen to fit.

Agent 3 must be able to reconstruct a candidate from:

```text
candidate config
model_contract_v2
architecture family version
initialization seed
```

## Deterministic construction

For each candidate require:

- same seed -> bit-identical CPU initial state dict;
- different seed -> weights differ;
- exact parameter count reproducible;
- config serialization round-trips;
- architecture digest stable;
- deterministic evaluation-mode CPU forward.

Use one declared initialization seed for the Phase 6 benchmark family unless a later agent explicitly needs a narrow sensitivity check.

## Basic smoke validation

For each candidate:

- construct on CPU;
- run a valid forward pass;
- verify exact output shapes;
- verify finite outputs;
- perform one small backward-connectivity smoke check;
- attempt MPS float32 construction/forward;
- attempt MPS float16 construction/forward where supported;
- perform a checkpoint/config save-load smoke test under `model_contract_v2`.

Do not run large batch sweeps; Agent 3 owns them.

## Parameter accounting

Record per candidate:

```text
exact trainable parameters
float32 parameter bytes
float16 parameter bytes
representative checkpoint size
encoder parameters
policy-head parameters
value-head parameters
belief-head parameters
```

## Files you own

Suggested:

```text
stratego/model/architecture_configs.py
stratego/model/production_model.py
tests/model/test_architecture_family.py
scripts/run_phase6_agent02.py
reports/phase_6_data/agent_02_architecture_family.json
```

Preserve `integration_model_v1` as the Phase 5 fixture unless repository conventions strongly favor a non-semantic relocation.

## Data file

Minimum contents:

```text
agent
status
architecture_family_version
initialization_seed
candidate_table
candidate_configs
parameter_counts
parameter_breakdowns
checkpoint_bytes
cpu_forward_results
mps_float32_smoke
mps_float16_smoke
backward_smoke
determinism_checks
config_digests
test_total
test_passed
test_failed
files_created
files_modified
completion_gates
```

## Completion gate

PASS only if:

- Agent 1 PASS verified;
- pre-existing suite green;
- one configurable family implements all candidates;
- C0-C6 configs are explicit/serializable;
- construction is deterministic;
- exact parameter counts recorded;
- policy/value/belief outputs match `model_contract_v2`;
- policy logits remain perspective-normalized;
- no privileged inputs exist;
- CPU smoke checks pass for all constructible candidates;
- MPS float32 smoke passes for every candidate intended for Agent 3;
- MPS float16 is honestly attempted;
- backward connectivity passes for viable candidates;
- checkpoint/config mismatch rejection works;
- full suite green.

## Handoff notes for Agent 3

Provide:

```text
exact candidate config table
model-construction API
architecture-family version
initialization seed
parameter counts
config digests
canonical valid benchmark-input builder
output validation API
basic MPS limitations, if any
```

Agent 3 must benchmark these exact candidates without architecture edits.
