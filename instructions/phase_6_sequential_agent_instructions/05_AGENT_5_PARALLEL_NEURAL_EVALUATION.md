# Phase 6 Agent 5 — Checkpoint-Aware Parallel Neural Evaluation

## Role

You are **Agent 5** in a sequential Phase 6 implementation for the Stratego project.

Work from the **root of the Stratego repository**.

Complete only the work assigned in this file. Do not begin Agent 6's architecture decision.

## Frozen project contracts

Do not alter:

- Phase 4 MatchSpec/match IDs;
- paired-unit IDs;
- `evaluation_setup_bank_v1`;
- `color_swap_same_board`;
- policy/decision seeds;
- match result/replay semantics;
- paired bootstrap statistics;
- baseline policy behavior;
- Agent 1 `model_contract_v2`.

Transport/parallelism must be invisible to match identity.

## Required reading

Read:

```text
00_PHASE_6_SEQUENCE_AND_COMMON_CONTRACT.md
reports/phase_4_implementation_report.md
reports/phase_4_data/
reports/phase_5_implementation_report.md
reports/phase_6_implementation_report.md
reports/phase_6_data/agent_04_finalists.json
stratego/evaluation/
stratego/model/policy_adapter.py
```

## Shared reporting contract

Append only:

```markdown
## 5. Agent 5 — Checkpoint-Aware Parallel Neural Evaluation
```

Create:

```text
reports/phase_6_data/agent_05_parallel_neural_evaluation.json
```

## Stop conditions

Mark `BLOCKED` if:

- Agents 1-4 are not PASS;
- deterministic neural evaluation cannot be preserved without changing MatchSpec/seed semantics;
- safe MPS ownership cannot be achieved;
- privileged state is required;
- parallel transport changes Phase 4 result/statistical semantics;
- deterministic greedy action histories cannot be made independent of worker count.

Do not weaken reproducibility to make parallelism pass.

## Prerequisite

Agents 1-4 must report `PASS`.

Use one stable Agent 4 finalist as the test checkpoint/policy.

Verify its exact model contract/config/checkpoint metadata before implementation.

Run the existing suite before edits.

## Objective

Remove Phase 5's serial-neural-evaluation limitation.

Build a checkpoint-aware parallel evaluation path in which the neural model is loaded once per long-lived inference owner and many matches can use it without changing deterministic game results.

## MPS ownership requirement

Do not casually spawn one MPS model per CPU game worker.

Prefer:

```text
CPU game workers
-> observer-safe inference requests
-> one long-lived MPS inference coordinator/service
-> checkpoint loaded once
-> batched model inference
-> deterministic action responses
-> games continue
```

If a different topology is chosen, justify its Apple MPS safety and prove checkpoint/model loading does not occur per game or move.

Required normal-run property:

```text
checkpoint loads per long-lived inference owner = 1
```

## Observer-safe payload

Transport only products the neural policy legitimately needs, such as:

```text
match/request identity
decision seed
acting player if needed at the conversion boundary
observation_v2_1_127ch
legal action product
```

Never transport:

- `GameState`;
- `PieceRecord`;
- hidden true identities;
- privileged belief targets;
- true opponent setup;
- privileged replay object.

Add an interface/object-graph regression.

## Deterministic batching

Worker count and arrival timing must not change games.

Use a deterministic request ordering/batching design, or another explicitly proven protocol.

Do not assume approximate float batch equivalence guarantees identical actions in near-tie positions.

The same MatchSpecs must yield the same relevant inference/action sequence independent of:

```text
worker count
chunking
arrival timing
schedule input order
```

Seeded categorical evaluation must continue to use the per-decision Phase 4 seed. Do not consume a global random stream in arrival order.

## Greedy reproducibility sweep

Run the same stored deterministic-greedy schedule with:

```text
1 worker
2 workers
4 workers
8 workers
8 workers, schedule input shuffled
```

Require identical:

```text
match IDs
paired-unit IDs
setups
policy seeds
decision seeds
absolute action histories or replay digests
winners
terminal reasons
ply counts
statistics
```

Required:

```text
1 distinct results digest
1 distinct replay-digest set
0 field-level mismatches
```

Use enough matches to exercise both colors and multiple baseline opponents.

## Seeded categorical reproducibility

Run a smaller meaningful seeded-categorical schedule at multiple worker counts plus shuffled input.

Require identical results from identical MatchSpecs/seeds.

Also prove this stochastic path differs from greedy on at least some decisions/results so the test is not accidentally executing the greedy branch.

## Failure behavior

Test at least:

- missing checkpoint;
- incompatible checkpoint;
- corrupted checkpoint;
- inference coordinator failure;
- malformed request;
- non-finite model output;
- normalized selection converting to an illegal absolute action;
- timeout/disconnect behavior if the design includes it.

Never substitute:

```text
random legal
first legal
previous action
```

on failure.

Preserve Phase 4 fail-fast/quarantine semantics.

## Performance

Measure:

```text
checkpoint load count
checkpoint load time
worker count
matches/second
positions/second if available
neural inference batch sizes
MPS utilization
request queue wait
CPU worker utilization
process memory
Metal memory
```

Verify the checkpoint is not loaded once per game/move.

Correctness is more important than maximum speed.

## Files you own

Suggested:

```text
stratego/evaluation/neural_worker.py
scripts/run_phase6_agent05.py
tests/evaluation/test_parallel_neural_checkpoint.py
reports/phase_6_data/agent_05_parallel_neural_evaluation.json
```

Use repository conventions if another additive layout is cleaner.

## Data file

Minimum contents:

```text
agent
status
topology
checkpoint
model_contract
checkpoint_load_counts
observer_safe_payload
greedy_worker_sweep
greedy_results_digests
greedy_replay_digests
greedy_field_mismatches
seeded_worker_sweep
seeded_reproducibility
failure_tests
throughput
memory
information_safety
test_total
test_passed
test_failed
files_created
files_modified
completion_gates
```

## Tests

Add tests for:

- neural inference owner lifecycle;
- checkpoint loaded once;
- observer-safe request graph;
- deterministic batching/request ordering;
- worker-count independence;
- schedule-shuffle independence;
- per-decision seed preservation;
- failure propagation;
- no legal-move substitution.

Run the full suite.

## Completion gate

PASS only if:

- Agents 1-4 PASS verified;
- stable finalist checkpoint loads under v2;
- checkpoint loaded once per long-lived inference owner;
- MPS ownership topology safe/documented;
- Phase 4 identities/seeds unchanged;
- observer-safe payload only;
- greedy 1/2/4/8/shuffled sweep has 0 mismatches;
- one results digest and one replay-digest set;
- seeded categorical mode reproduces;
- failure cases are loud and never substitute a move;
- throughput/load overhead measured;
- full suite green.

## Handoff notes for Agent 6

Provide the stable checkpoint-aware parallel evaluation API and any measured throughput/memory limitation.

Agent 6 must not need to redesign evaluation while selecting the architecture.
