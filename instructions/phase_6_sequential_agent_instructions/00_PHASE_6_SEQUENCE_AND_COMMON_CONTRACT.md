# Phase 6 Sequential Agent Plan

## Goal

Select the first real Stratego move-network architecture for the M4 Pro by combining:

- a corrected perspective-normalized model action contract;
- a controlled Transformer candidate family;
- standalone MPS inference and backward benchmarks;
- real Phase 3 self-play-pipeline measurements;
- deterministic parallel neural evaluation;
- a one-hour production soak and exact 168-hour compute/storage projection.

Phase 6 should answer:

- Does the normalized action-frame contract preserve all frozen engine/evaluation semantics?
- What model sizes are numerically stable on Apple MPS?
- Where is the M4 Pro compute/capacity knee?
- Which candidates remain practical once inserted into the real self-play pipeline?
- Can neural checkpoints be evaluated in parallel without changing games?
- Which exact architecture should be the primary training model?
- What exact smaller architecture should be the fallback?
- What will the selected model likely produce during the user's 168-hour final run?

## Agent sequence

| Agent | Task |
|---|---|
| 1 | `model_contract_v2`, perspective-normalized actions, compatibility and safety validation |
| 2 | Configurable candidate Transformer family C0-C6 |
| 3 | Standalone MPS inference/backward benchmark and shortlist |
| 4 | Integrated Phase 3 collection/recording benchmark and finalists |
| 5 | Checkpoint-aware deterministic parallel neural evaluation |
| 6 | One-hour soak, 168-hour projection, primary/fallback architecture recommendation |

Run strictly in order.

## Shared report

```text
reports/phase_6_implementation_report.md
```

Owned sections:

```text
# Phase 6 Implementation Report

## 1. Agent 1 — Model Contract v2 and Perspective-Normalized Actions
## 2. Agent 2 — Candidate Architecture Family
## 3. Agent 3 — Standalone MPS and Training-Step Benchmark
## 4. Agent 4 — Integrated Self-Play Pipeline Benchmark
## 5. Agent 5 — Checkpoint-Aware Parallel Neural Evaluation
## 6. Agent 6 — Stability Soak, 168-Hour Projection, and Architecture Decision
```

Agent 1 creates the report header if absent. Later agents append only their own section.

## Canonical data files

```text
reports/phase_6_data/agent_01_model_contract_v2.json
reports/phase_6_data/agent_02_architecture_family.json
reports/phase_6_data/agent_03_inference_benchmark.csv
reports/phase_6_data/agent_03_training_step_benchmark.csv
reports/phase_6_data/agent_03_architecture_shortlist.json
reports/phase_6_data/agent_04_integrated_pipeline.csv
reports/phase_6_data/agent_04_storage_rates.csv
reports/phase_6_data/agent_04_finalists.json
reports/phase_6_data/agent_05_parallel_neural_evaluation.json
reports/phase_6_data/agent_06_soak.json
reports/phase_6_data/agent_06_soak_timeseries.csv
reports/phase_6_data/agent_06_weekly_projection.json
reports/phase_6_data/agent_06_architecture_decision.json
```

Optional raw CSV/JSON files may be added with the same `agent_0X_` prefix.

## Frozen project state

```text
rules                       stratego_project_v1
reference engine            phase2_1_reference_1.1.0
observation                 observation_v2_1_127ch
engine action encoding      source_destination_10000_v1
engine action frame         absolute engine squares
Phase 3 backend             KEEP_PYTHON
Phase 4 evaluation          COMPLETE
Phase 4 setup bank          evaluation_setup_bank_v1
Phase 4 pairing             color_swap_same_board
Phase 5 integration         COMPLETE
Phase 5 model contract      model_contract_v1
Phase 5 checkpoint format   version 1
Phase 5 fixture             integration_model_v1 — integration only, not production
```

Accepted Phase 5 starting evidence should be verified from the real repository:

```text
full suite                  2,155 passed, 2 deliberate skips, 0 failed
action audit                10,000/10,000, 0 mismatches
neural hidden-info audit    10,000 valid trials, 0 mismatches
checkpoint negative cases   24/24 rejected
MPS float32                 PASS
MPS float16                 PASS
batch equivalence           PASS
Phase 4 neural gauntlet     1,024 matches, 0 illegal, 0 policy failures
```

## Approved Phase 6 action-frame decision

The model-facing action space is now **perspective-normalized**.

The engine remains absolute:

\[
a_{\mathrm{engine}}=100s_{\mathrm{absolute}}+d_{\mathrm{absolute}}.
\]

The model uses:

\[
a_{\mathrm{model}}=100s_{\mathrm{normalized}}+d_{\mathrm{normalized}}.
\]

The intended boundary is:

```text
absolute engine legal actions/mask
        ↓
perspective conversion
        ↓
normalized model legal actions/mask
        ↓
model selects normalized action
        ↓
inverse perspective conversion
        ↓
absolute engine action
        ↓
independent engine validation
```

Do **not** change the engine's 10,000 absolute action identifiers.

## Common model semantics

After Agent 1 passes, all Phase 6 candidates use:

```text
input                [B, 127, 10, 10]
tokens               [B, 100, 127]
token frame          perspective-normalized row-major
policy logits        [B, 10000]
policy frame         perspective-normalized source/destination
value logits         [B, 3]
value order          WIN, DRAW, LOSS
value perspective    acting/model player
belief logits        [B, 100, 12]
belief supervision   unresolved hidden opponent pieces only
```

Privileged belief targets remain training-side labels and never model inputs.

## Phase 6 candidate family target

Use one configurable architecture family, initially:

| ID | Width | Blocks | Heads | Feed-forward |
|---|---:|---:|---:|---:|
| C0 | 64 | 2 | 4 | 256 |
| C1 | 128 | 4 | 4 | 512 |
| C2 | 192 | 4 | 6 | 768 |
| C3 | 192 | 6 | 6 | 768 |
| C4 | 256 | 6 | 8 | 1,024 |
| C5 | 256 | 8 | 8 | 1,024 |
| C6 | 384 | 8 | 8 | 1,536 |

C6 is an upper-region reference, not a presumed final choice.

## General correctness rules

- Do not modify `stratego/engine/`.
- Do not alter rules, combat, terminal precedence, hidden-information semantics, observation channels, replay semantics, or absolute engine action IDs.
- Do not alter Phase 4 setup bank, MatchSpec identity, pairing, decision seeds, result semantics, or paired bootstrap.
- Never pass `GameState`, true hidden identities, opponent setup truth, privileged replay objects, or belief targets into a model/policy.
- The engine remains the final legality authority.
- Never substitute a legal move after a policy/model failure.
- Required MPS measurements must actually use MPS; never silently substitute CPU.
- Preserve prior tests. Any newly discovered implementation bug requires a regression test.
- Do not silently weaken acceptance checks.
- Random-weight playing strength must not influence architecture selection.
- No reinforcement learning, warm-start training, self-play learning, optimizer tuning, learning-rate tuning, or hyperparameter search is authorized in Phase 6.
- Benchmark backward passes are allowed only to measure compute and verify gradient connectivity.
- Do not edit `stratego_project_docs/` during Phase 6.
- Do not begin Phase 7.

## Phase 3 reference measurements

The Phase 3 representative probe was not a production model. These values are reference points only and must be re-measured with real candidates:

```text
simulation numerator, no model       ~96,963 positions/s
representative model denominator     ~14,922 positions/s
R                                    6.50
integrated no-recording best         ~12,838 positions/s
recording soak                       ~8,871 positions/s
trajectory production                ~5.59 GiB/hour
snapshot interval                    32
starting workers                     10
starting environments                1,536
live legality                        dense
backend                              KEEP_PYTHON
```

## Shared reporting contract

Each agent section must contain:

1. `PASS`, `FAIL`, or `BLOCKED`;
2. prerequisite verification;
3. implementation/measurement summary;
4. files created/modified;
5. tests/commands run;
6. measured results;
7. completion-gate table;
8. deviations/limitations;
9. data-file paths;
10. handoff notes.

Every headline number in Markdown must also exist in a machine-readable artifact.

Every primary JSON should record at least:

```text
agent
phase
status
timestamp
commit
platform
python_version
torch_version
mps_built
mps_available
prerequisite_status
tests_before
tests_after
commands
durations
seeds
files_created
files_modified
completion_gates
problems
```

## Stop conditions

Mark `BLOCKED` and stop if:

- a prerequisite agent did not pass;
- frozen engine/evaluation semantics would need modification;
- privileged hidden information appears necessary;
- the pre-existing suite is unexpectedly red and the failure is not clearly environmental;
- action-frame conversion is ambiguous or non-bijective;
- a required MPS correctness measurement cannot actually use MPS;
- satisfying one acceptance condition necessarily violates another frozen guarantee.

For an ordinary bug in new Phase 6 code, fix it, add a regression test, rerun relevant and full suites, and document it.

## Global Phase 6 acceptance

Phase 6 can be recommended `PASS` only if:

- `model_contract_v2` safely implements normalized model actions;
- the candidate family is reproducible and uses common semantics;
- standalone inference/backward benchmarks are fair and reproducible;
- real integrated collection and production recording are measured;
- no illegal actions or reconstruction mismatches occur in headline runs;
- deterministic parallel neural evaluation reproduces across worker counts;
- one finalist completes the one-hour stability soak;
- one exact primary architecture is selected;
- one exact smaller fallback architecture is selected;
- the decision is justified by the measured compute/capacity frontier;
- a 168-hour compute/storage projection is produced;
- storage is analyzed against ~150 GB internal and ~1 TB external capacity;
- the full repository suite remains green;
- no random-weight playing-strength claim is used as evidence.
