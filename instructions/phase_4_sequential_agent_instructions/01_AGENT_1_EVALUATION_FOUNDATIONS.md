# Phase 4 Agent 1 — Evaluation Foundations and Setup Bank

## Role

You are **Agent 1** in a sequential Phase 4 implementation for the Stratego project.

Work from the **root of the Stratego repository**.

Complete only the work assigned in this file. Do not begin later-agent tasks.

## Frozen project contracts

Do not alter:

- reference engine: `phase2_1_reference_1.1.0`
- rules: `stratego_project_v1`
- observation: `observation_v2_1_127ch`
- action encoding: 10,000 source-destination action identifiers
- Phase 3 production simulator decision: `KEEP_PYTHON`

The Python reference engine remains the behavioral authority.

Do not modify game rules, observation semantics, action encoding, combat behavior,
terminal precedence, replay semantics, or hidden-information rules.

## Phase 4 scope

Phase 4 is:

> **Baseline opponents and evaluation harness**

It is **not**:

- neural-network training;
- final model architecture selection;
- setup-generator Phase 7 work;
- self-play reinforcement learning;
- belief-model training;
- decision-time search;
- browser/interface work;
- optimized-engine work.

The untrained Phase 3 representative Transformer is a benchmark probe and must
not be treated as a real Stratego opponent.

## Required reading

Before implementation, read at least:

```text
stratego_project_docs/README.md
stratego_project_docs/02_project_ruleset.md
stratego_project_docs/03_game_engine_spec.md
stratego_project_docs/04_engine_validation_plan.md
stratego_project_docs/05_project_plan.md
stratego_project_docs/06_observation_v2_127ch.md
stratego_project_docs/08_internal_state_spec.md
stratego_project_docs/09_public_event_and_replay_schema.md
reports/phase_3_implementation_report.md
```

Also inspect all prior code/tests your task depends on.

## Shared reporting contract

All Phase 4 agents append to one Markdown report:

```text
reports/phase_4_implementation_report.md
```

Each agent owns one numbered top-level section and must not rewrite earlier
sections.

All raw/machine-readable results go under:

```text
reports/phase_4_data/
```

Each agent must use its own filenames and never overwrite another agent's data.

Each report section must contain:

1. `PASS`, `FAIL`, or `BLOCKED`;
2. implementation summary;
3. files created/modified;
4. tests run;
5. measured results;
6. deviations/limitations;
7. data-file paths;
8. handoff notes.

## General correctness rules

- Policies may use only information legal for the acting player.
- Privileged `GameState` fields and belief targets must never enter a policy.
- Every returned action must be legal.
- The engine remains the final legality authority.
- Randomness must be reproducible from explicit seeds.
- Match identity must not depend on worker count or execution order.
- Hidden-information permutation tests must fail loudly on any unexplained policy change.
- Do not silently weaken tests to make a result pass.

## Stop conditions

Mark `BLOCKED` and stop if:

- a prerequisite agent did not pass;
- frozen engine semantics would need modification;
- policy code requires privileged hidden information;
- deterministic reproduction cannot be maintained;
- an evaluator cannot reproduce the same scheduled match from its identifiers;
- you discover an unresolved engine correctness issue.

Do not fix frozen-engine semantics yourself.

## Prerequisite

Phase 3 must be complete with:

```text
backend decision = KEEP_PYTHON
```

Confirm the existing repository test suite is green before implementation.

## Objective

Create the policy/evaluation contracts that all later Phase 4 agents will use.

This agent owns:

- observer-safe policy input contract;
- policy result contract;
- deterministic match identity;
- evaluation setup-bank schema;
- fixed evaluation-only setup bank;
- paired match scheduling primitives.

Do not implement baseline strategy logic beyond a minimal test policy.

## Files you own

Suggested layout:

```text
stratego/evaluation/__init__.py
stratego/evaluation/policy.py
stratego/evaluation/match_spec.py
stratego/evaluation/setup_bank.py
tests/evaluation/__init__.py
tests/evaluation/test_policy_contract.py
tests/evaluation/test_match_spec.py
tests/evaluation/test_setup_bank.py
scripts/run_phase4_agent01.py
reports/phase_4_data/agent_01_evaluation_foundations.json
```

Small helper modules are allowed if clearly justified.

## Policy interface

Create one common policy interface usable by:

- Phase 4 baselines;
- future neural checkpoints;
- later search policies.

A policy decision request may contain only observer-legal data.

At minimum support:

```text
policy_id
game_id / match identity
ply
acting player
observation_v2_1_127ch
legal action identifiers
optional public metadata that is already represented legally
policy random seed / random stream
```

The policy must not receive:

- privileged `GameState`;
- opponent true hidden identities;
- belief target;
- privileged replay;
- any engine object that permits reading hidden types.

The returned result must include at minimum:

```text
selected_action_id
policy_id
decision seed / reproducibility metadata
optional diagnostics
```

Selected action must be legal.

## Match identity contract

Define a deterministic match identifier from explicit components such as:

```text
evaluation suite version
candidate policy id/version
opponent policy id/version
setup pair id
color assignment
replicate index
root evaluation seed
```

The same match identifier must always imply the same:

- setups;
- colors;
- policy seeds;
- first player;
- rules configuration.

Changing worker count or execution order must not alter a match.

## Paired evaluation unit

Represent paired matches explicitly.

One paired unit should normally contain:

```text
Game A:
candidate as Red
opponent as Blue

Game B:
candidate as Blue
opponent as Red
```

Use equivalent/fixed setup conditions as defined by the setup-bank contract.

Do not assume that simply swapping colors without respecting setup orientation is equivalent. Define the exact transformation or pairing rule.

## Fixed evaluation setup bank

Create a deterministic, versioned evaluation-only bank.

Target:

- preferably **1,024 legal setup pairs**;
- minimum acceptable if runtime becomes unreasonable: **512 pairs**.

This bank is for reproducible evaluation only.

It must not become the Phase 7 training setup generator.

For every entry record:

```text
setup_pair_id
red setup
blue setup
generation seed / provenance
bank version
```

Validate:

- exact legal inventory;
- legal setup rows;
- no overlap/lakes;
- deterministic regeneration;
- unique pair identifiers.

Use simple deterministic generation with enough variation to avoid one single arrangement repeated with superficial permutations.

Do not create learned setup logic here.

## Reproducibility tests

At minimum verify:

- same setup-bank seed -> byte-identical bank;
- same match spec -> same derived policy seeds;
- schedule order does not affect match identities;
- shuffling match list does not change individual specifications;
- worker-index assignment is not included in match identity;
- paired color unit reconstructs exactly.

## Data file

Write:

```text
reports/phase_4_data/agent_01_evaluation_foundations.json
```

Minimum contents:

```text
agent
status
policy_interface_version
match_spec_version
setup_bank_version
setup_pair_count
setup_validation_failures
duplicate_setup_pair_ids
determinism_trials
determinism_failures
paired_units_tested
paired_unit_failures
test_total
test_passed
test_failed
files_created
files_modified
```

## Shared report section

Create `reports/phase_4_implementation_report.md` if absent.

Header:

```markdown
# Phase 4 Implementation Report

Frozen reference: `phase2_1_reference_1.1.0`  
Rules: `stratego_project_v1`  
Observation: `observation_v2_1_127ch`  
Phase 3 backend: `KEEP_PYTHON`
```

Append:

```markdown
## 1. Agent 1 — Evaluation Foundations and Setup Bank
```

## Completion gate

PASS only if:

- existing tests remain green;
- policy API contains no privileged state;
- setup bank contains at least 512 valid deterministic setup pairs;
- match identities are deterministic;
- paired scheduling is explicit and reproducible;
- all new tests pass.

Do not implement the real baseline suite.
