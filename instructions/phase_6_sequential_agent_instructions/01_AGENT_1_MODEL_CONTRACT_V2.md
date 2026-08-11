# Phase 6 Agent 1 — Model Contract v2 and Perspective-Normalized Actions

## Role

You are **Agent 1** in a sequential Phase 6 implementation for the Stratego project.

Work from the **root of the Stratego repository**.

Complete only the work assigned in this file. Do not begin later-agent tasks.

## Frozen project contracts

Do not alter:

- reference engine: `phase2_1_reference_1.1.0`;
- rules: `stratego_project_v1`;
- observation: `observation_v2_1_127ch`;
- engine action encoding: `source_destination_10000_v1`;
- Phase 3 backend: `KEEP_PYTHON`;
- Phase 4 match/evaluation semantics.

The engine continues to use absolute source/destination square identifiers.

## Phase 6 scope

Phase 6 is:

> **Production model architecture selection and M4 Pro benchmarking**

Your part is only the model-contract migration and its correctness/safety validation.

Do not:

- build the candidate architecture family;
- benchmark model sizes;
- train a model;
- modify `stratego/engine/`;
- change Phase 4 identities/statistics.

## Required reading

Before implementation, read at least:

```text
00_PHASE_6_SEQUENCE_AND_COMMON_CONTRACT.md
stratego_project_docs/README.md
stratego_project_docs/02_project_ruleset.md
stratego_project_docs/03_game_engine_spec.md
stratego_project_docs/05_project_plan.md
stratego_project_docs/06_observation_v2_127ch.md
reports/phase_5_implementation_report.md
reports/phase_5_data/
stratego/model/
stratego/engine/actions.py
stratego/evaluation/
```

Also inspect all existing tests your task depends on.

## Shared reporting contract

Append only:

```markdown
## 1. Agent 1 — Model Contract v2 and Perspective-Normalized Actions
```

to:

```text
reports/phase_6_implementation_report.md
```

Write machine-readable results to:

```text
reports/phase_6_data/agent_01_model_contract_v2.json
```

Your report section must contain status, implementation summary, files, tests, measured results, deviations, data paths, completion gates, and Agent 2 handoff notes.

## General correctness rules

- The engine remains the final legality authority.
- Model inputs remain observer-safe only.
- The model must never see true hidden identities or belief targets.
- Randomness remains explicitly seeded.
- A v1 checkpoint must never be silently reinterpreted under v2.
- Do not weaken Phase 5 tests to make the migration pass.

## Stop conditions

Mark `BLOCKED` if:

- Phase 5 was not accepted;
- the current suite is unexpectedly red before edits;
- normalized actions require changing engine action IDs;
- the engine's existing perspective helpers are internally inconsistent;
- Phase 4 results would require a semantic schema change;
- hidden-information safety breaks.

## Prerequisite

Phase 5 must be formally accepted.

Verify from the real repository/report:

```text
22 / 22 Phase 5 hard gates true
model_contract_v1 present
integration_model_v1 marked integration-only
```

Run the full repository suite before implementation and record exact totals.

## Objective

Create a new explicit model contract, expected to be:

```text
model_contract_v2
```

with:

```text
TOKEN_SQUARE_FRAME     perspective_normalized_squares
POLICY_ACTION_FRAME    perspective_normalized_squares
ENGINE_ACTION_FRAME    absolute_engine_squares
```

The engine action encoding remains:

```text
source_destination_10000_v1
```

## Action-frame transformations

Use one authoritative model-layer conversion implementation, preferably reusing the frozen engine's perspective-coordinate helpers.

Support the conceptual operations:

```text
absolute_action_to_model(action_id, acting_player)
model_action_to_absolute(action_id, acting_player)
absolute_legal_actions_to_model(...)
absolute_legal_mask_to_model(...)
```

For Red, mapping is identity.

For Blue, each square uses the existing 180-degree player-relative normalization:

```text
square -> 99 - square
```

Both source and destination transform.

Do not create a second competing coordinate convention.

## Exhaustive action audit

For every action identifier `0..9999` and both players, require:

```text
absolute -> model -> absolute
```

returns the original action exactly.

Required cases:

```text
10,000 actions × 2 players = 20,000
```

Also validate:

```text
model -> absolute -> model
```

over all 20,000 player/action combinations.

Require:

- zero mismatches;
- no collisions;
- full bijection.

Pin representative geometry including boundaries, long Scout-style displacements, lateral/vertical moves, and first/last rows and columns.

## Legal-action and dense-mask equivalence

Generate a broad corpus of real nonterminal positions over both colors and multiple plies.

For each position:

1. obtain the engine's absolute legal-action list and dense mask;
2. transform both into model frame;
3. require transformed list equals nonzero transformed-mask indices;
4. map the normalized legal set back to absolute;
5. require exact equality with the original engine legal set.

Zero mismatches.

Do not hard-code an assumed maximum legal-action count.

## Policy adapter migration

The neural decision path must become:

```text
normalized observation
-> model logits in normalized action frame
-> normalized engine-derived legality
-> greedy or seeded categorical selection
-> selected normalized action
-> model_action_to_absolute(...)
-> PolicyResult with absolute engine action
-> Phase 4 validation
-> engine apply_action
```

Preserve:

- deterministic greedy tie-break;
- one random stream per decision;
- seeded categorical reproducibility;
- non-finite rejection;
- malformed/empty legality rejection;
- no action substitution;
- independent engine validation.

## Checkpoint compatibility

Update checkpoint metadata/validation so:

- v2 records the normalized policy frame;
- v1 absolute-frame checkpoints fail under v2;
- v2 checkpoints fail under v1 semantics;
- missing/wrong action-frame metadata fails loudly;
- semantic checks happen before misleading tensor-shape errors where practical.

Do not mutate old checkpoint metadata on load.

## Symmetry regression

Construct color-swapped, 180-degree-rotated equivalent positions.

Require equivalent roles receive:

```text
identical normalized observations
identical normalized legal masks
```

Use a deterministic model or crafted logits and require the same **normalized strategic action** in both positions.

The resulting absolute engine actions should be the corresponding rotated moves.

This is the core learning-symmetry benefit of v2.

## Hidden-information audit

Run at least:

```text
10,000 valid model-level hidden-information permutation trials
```

under v2.

Require zero differences in:

- normalized observation;
- normalized legal set/mask;
- policy logits;
- value logits;
- belief logits;
- model-frame chosen action;
- absolute engine chosen action;
- public diagnostics.

Positive controls must prove:

- hidden true types changed;
- privileged belief targets changed.

Zero positive-control failures.

## Phase 4 integration regression

Run a defensible neural evaluation subset in both policy modes and both colors.

Require:

- 0 illegal actions;
- 0 policy failures;
- deterministic greedy rerun;
- reproducible seeded-categorical rerun;
- absolute replay/action histories remain correct;
- MatchSpec/result semantics unchanged.

Playing strength is irrelevant.

## Files you own

Prefer changes/additions under:

```text
stratego/model/
tests/model/
scripts/run_phase6_agent01.py
reports/phase_6_data/agent_01_model_contract_v2.json
```

Minimal additive evaluation wiring is allowed only if needed.

Do not modify `stratego/engine/`.

## Data file

Minimum JSON fields:

```text
agent
status
model_contract_version
token_square_frame
policy_action_frame
engine_action_frame
action_round_trip_cases
action_round_trip_mismatches
reverse_round_trip_cases
reverse_round_trip_mismatches
legal_positions_tested
legal_action_comparisons
legal_mask_mismatches
symmetry_trials
symmetry_mismatches
hidden_information_trials
hidden_information_mismatches
positive_control_failures
checkpoint_compatibility_tests
checkpoint_rejection_failures
evaluation_regression
test_total
test_passed
test_failed
files_created
files_modified
completion_gates
```

## Completion gate

PASS only if:

- Phase 5 accepted;
- pre-existing suite green;
- `model_contract_v2` explicit/versioned;
- engine action semantics unchanged;
- 20,000 absolute round trips have 0 mismatches;
- 20,000 reverse round trips have 0 mismatches;
- real-position legal list/mask conversion is exact;
- symmetry regression passes;
- v1/v2 incompatible checkpoints fail loudly;
- >=10,000 hidden-information trials have 0 mismatches;
- positive controls all succeed;
- greedy and seeded modes reproduce;
- Phase 4 integration regression has 0 illegal/policy/replay mismatches;
- full suite green.

## Handoff notes for Agent 2

Document the exact public APIs Agent 2 must use for:

```text
model_contract_v2
tokenization
normalized policy logits
normalized legality
checkpoint metadata
model-output validation
```

Agent 2 must not invent another action frame or duplicate conversion logic.
