# Phase 8 Agent 3 — Training Examples, Targets, and Anti-Leak Audit

## Role

You are **Agent 3**.

Convert the accepted compact synthetic corpus into deterministic model-training examples and independently prove that policy/value/belief targets are correct and isolated from model inputs.

Do not run meaningful C1 training.

## Prerequisite

Agents 1–2 must be PASS.

Read their artifacts and verify corpus digests before work.

Run the full suite before edits.

## Mission

Implement:

```text
warmstart_example_v1
warmstart_decision_sampler_v1
```

and a streaming/indexed dataset that reconstructs selected examples without materializing all `127 x 10 x 10` observations on disk.

The dataset must support deterministic:

```text
train iteration
validation iteration
sealed test structural audit
resume from exact data cursor
```

## Decision-index generation

Independently reproduce Agent 1's exact max-64-per-game selection.

Create a stable selected-example identity:

```text
(game_id, decision_index)
```

The same corpus manifest and decision-sampler version must always yield the same ordered universe before shuffle.

Record split counts:

```text
selected examples total
policy-supervised examples
value-supervised examples
belief-supervised pieces
```

by policy, matchup, setup family, and game-progress bucket.

## Replay reconstruction

For each selected decision:

1. restore/replay the game to the pre-action state;
2. build observer-safe `observation_v2_1_127ch`;
3. obtain absolute engine legal actions/mask;
4. identify acting player;
5. obtain the recorded absolute action;
6. convert observation/legal/action to the model frame as required;
7. create WDL target;
8. build hidden-only belief targets from privileged truth after observation creation.

Use the existing action-frame conversion API. Do not duplicate its math.

## Policy target audit

Hard requirements:

```text
recorded action is in engine legal set
model-frame action inverse-converts to same absolute action
model-frame legal mask exactly matches converting the absolute legal set
policy weight matches acting policy id
random/stress weight == 0
strategic/tactical/basic weights exact
```

For at least:

```text
100,000 selected decisions or all if fewer
```

require zero mismatches.

For a deterministic sample of policy-supervised positions, re-invoke the original rule policy with its recorded seed/context where feasible and require the recorded action to reproduce.

Target at least:

```text
10,000 direct teacher-decision reproductions
0 mismatches
```

If a Phase 4 policy's seed contract makes direct local reinvocation impossible, report `BLOCKED` rather than fabricating a weaker teacher proof.

## Value target audit

Independently map final game result to acting-player perspective.

Exhaustively validate selected examples if practical.

Required result mapping:

```text
acting player winner -> WIN
draw                 -> DRAW
opponent winner      -> LOSS
```

Inject negative tests for color inversion.

No model prediction enters the target.

## Belief target audit

For every supervised square:

```text
square currently contains an opponent piece
piece identity is unresolved to acting player
target class equals privileged true type
target square is model-perspective normalized
```

For every ignored square:

```text
empty/lake OR own piece OR revealed/known opponent
```

must not contribute to belief loss.

Test Red and Blue thoroughly.

## Anti-leak boundary

Build an explicit training-batch boundary.

The model forward call must receive:

```text
observation tensor only
```

No true hidden type, belief mask/label, opponent setup, setup family/base id, teacher id, teacher private score, game result, or future action may be reachable from the model input object.

Legal mask is allowed at the loss/adapter layer but is not an observation channel.

Use an object-graph/interface regression and a positive control.

## Hidden-permutation training test

Construct paired privileged states that have identical public information but permute unresolved opponent true identities.

Require:

```text
observations identical
legal masks identical
policy target/action-frame semantics unchanged when public action is fixed

belief targets change in the positive-control cases
privileged truth changes
```

Run at least:

```text
25,000 valid paired example trials
```

with zero unexplained model-input mismatch.

This is a training-pipeline anti-leak test, not a replacement for Phase 2's larger engine anti-leak validation.

## Baseline implementation

Implement Agent 1's frozen:

```text
uniform-legal policy baseline
train-fitted constant WDL value baseline
remaining-inventory marginal belief baseline
```

Fit the value prior from train selected examples only.

Generate and freeze validation baseline metrics.

Do **not** compute model metrics on the test split.

Test may be parsed only for structural target correctness.

## Dataset throughput

Benchmark reconstruction feeding with:

```text
1, 2, 4, 8, 10 CPU workers or practical subset
batch size 256
```

Measure:

```text
examples/s
p50/p95 batch construction latency
CPU utilization
peak RSS
snapshot seek/replay cost
```

The goal is to supply C1 faster than the MPS trainer consumes examples.

Phase 6 measured standalone C1 training around 3k examples/s; treat that only as a reference and measure the real Phase 8 pipeline.

## Deterministic shuffle / cursor

Implement a versioned deterministic train order.

Requirements:

```text
same epoch/order seed -> same selected-example order
resume cursor -> exact next example/batch
different worker count -> same logical batch identities
prefetch timing -> no ordering change
```

Do not use worker arrival order as the training order.

## Suggested files

```text
stratego/training/warmstart_examples.py
stratego/training/warmstart_dataset.py
stratego/training/warmstart_baselines.py

tests/training/test_warmstart_examples.py
tests/training/test_warmstart_targets.py
tests/information_security/test_warmstart_target_boundary.py

scripts/run_phase8_agent03.py
```

## Artifacts

Create:

```text
reports/phase_8_data/agent_03_example_contract.json
reports/phase_8_data/agent_03_target_audit.json
reports/phase_8_data/agent_03_validation_baselines.json
```

Append report section 3.

## PASS gates

PASS only if:

- Agent 2 corpus digests verified;
- deterministic selected-example universe exact;
- max 64 decisions/game contract exact;
- >=100,000 policy/action-frame target audits, 0 mismatches;
- >=10,000 direct teacher-decision reproductions, 0 mismatches;
- value perspective mapping exact;
- belief hidden-only semantics exact;
- 25,000 anti-leak paired trials clean with positive controls;
- model-input reachability contains no privileged target;
- validation baselines frozen;
- test model metrics not computed;
- deterministic shuffle/cursor exact;
- dataset throughput measured;
- full suite green.

## Handoff to Agent 4

Provide:

```text
example version/schema
dataset API
selected-example counts
train-order API
resume cursor API
baseline evaluators
validation baseline artifact
measured reconstruction throughput
anti-leak evidence
```
