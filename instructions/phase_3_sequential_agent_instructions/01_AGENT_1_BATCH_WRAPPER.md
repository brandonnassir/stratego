# Phase 3 Agent 1 — Batch Wrapper and Reference Equivalence

## Role and scope

You are **Agent 1** in a sequential Phase 3 implementation. Work only on the task in this file.

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

## Prerequisites

No Phase 3 prerequisite agent. First verify the existing Phase 2.1 suite passes.

Run:

```bash
python -m pytest -q
```

If the frozen Phase 2.1 suite fails before your changes, do not implement. Record `BLOCKED`.

## Objective

Build a correctness-first **single-process batch wrapper** around the frozen reference engine.

This agent does **not** implement multiprocessing, shared memory, MPS, or trajectory persistence.

The wrapper establishes the batch semantics later agents must preserve.

## Files you own

Create/adapt:

```text
stratego/training/__init__.py
stratego/training/batch_simulation.py
tests/training/test_batch_simulation.py
scripts/run_phase3_agent01.py
reports/phase_3_data/agent_01_batch_equivalence.json
```

You may add small helper files under `tests/training/` if clearly necessary.

Do not modify `stratego/engine/` unless it is only a nonsemantic import/export convenience and you can prove no behavior changed. Prefer not to modify it at all.

## Required batch API behavior

The wrapper must support many independent game slots and at minimum provide operations equivalent to:

- create `N` games;
- deterministically seed each slot;
- return each active game's acting player;
- return observations;
- return legal-action lists;
- return dense legal-action masks;
- apply one action per active game;
- expose terminal result/reason;
- independently reset selected/finished slots;
- preserve slot identity across resets.

Each slot must expose:

```text
environment_id
generation
```

`environment_id` remains fixed for the slot.

`generation` increments every time that slot is reset into a new game.

A trajectory or result must be identifiable by `(environment_id, generation)` so records cannot cross a reset boundary.

## Bulk-synchronous semantics

One batch step is conceptually:

```text
read all active observations/legal actions
-> choose one action per active environment
-> apply all chosen actions
-> collect terminal results
-> reset selected finished environments
-> next batch step
```

The implementation may loop over individual reference-engine states internally. Correctness is more important than speed here.

## Independent reset requirement

A finished environment must be reset without resetting other environments.

Test a batch containing games at substantially different plies and force only selected slots to terminate/reset.

Verify:

- unaffected slots are byte/fingerprint-equivalent before and after the other slot's reset;
- reset slot generation increments exactly once;
- reset slot begins a new legal game;
- no recent moves, behavioral events, counters, or knowledge from the prior generation survive.

## Differential-equivalence acceptance test

Compare the batch wrapper against independently stepped frozen reference games.

Use deterministic seeds.

Acceptance target:

- at least **100,000 state/action comparisons**;
- include ordinary moves, Scout moves, combat, reveals, all five behavioral event types when naturally encountered, terminal states, and resets;
- compare after every tested action:
  - full state fingerprint if available;
  - acting player;
  - legal-action list;
  - dense legal mask;
  - both players' observations;
  - public views/events where available;
  - terminal reason/result;
- **zero unexplained mismatches**.

Also verify that an illegal action rejected through the batch API cannot partially mutate a slot.

## Data file schema

Write:

```text
reports/phase_3_data/agent_01_batch_equivalence.json
```

Include at minimum:

```text
agent
status
implementation_version
rules_version
observation_version
python_version
test_total
test_passed
test_failed
batch_sizes_tested
state_action_comparisons
equivalence_mismatches
independent_reset_trials
reset_mismatches
generation_errors
illegal_action_inert_trials
illegal_action_inert_failures
behavior_types_observed
terminal_reason_counts
elapsed_seconds
files_created
files_modified
```

Raw per-case data is unnecessary unless a mismatch occurs. If a mismatch occurs, write a separate uniquely named reproduction JSON.

## Report section

Create the report if it does not exist:

```text
reports/phase_3_implementation_report.md
```

Start it with:

```markdown
# Phase 3 Implementation Report

Frozen reference: `phase2_1_reference_1.1.0`  
Rules: `stratego_project_v1`  
Observation: `observation_v2_1_127ch`
```

Then append only:

```markdown
## 1. Agent 1 — Batch Wrapper and Reference Equivalence
```

Summarize the evidence and link/path-reference your JSON data file.

## Completion gate

PASS only if:

- all existing Phase 2.1 tests still pass;
- all new batch tests pass;
- at least 100,000 equivalence comparisons have zero unexplained mismatches;
- independent reset and generation semantics pass;
- illegal-action inertness passes.

Do not implement multiprocessing.
