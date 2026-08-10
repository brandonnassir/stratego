# Phase 4 Agent 2 — Baseline and Stress Opponents

## Role

You are **Agent 2** in a sequential Phase 4 implementation for the Stratego project.

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

Agent 1 must be `PASS`.

Read:

```text
reports/phase_4_implementation_report.md
reports/phase_4_data/agent_01_evaluation_foundations.json
```

If not PASS, append Section 2 as `BLOCKED` and stop.

## Objective

Implement a baseline-policy suite using Agent 1's observer-safe policy interface.

This agent owns policy logic only.

Do not build the parallel league runner or statistical reporting beyond what is needed for unit tests.

## Files you own

Suggested layout:

```text
stratego/evaluation/baselines.py
stratego/evaluation/heuristics.py
tests/evaluation/test_baselines.py
tests/evaluation/test_baseline_information_safety.py
scripts/run_phase4_agent02.py
reports/phase_4_data/agent_02_baseline_agents.json
```

You may split baseline policies into multiple modules if it improves clarity.

## Required baseline policies

Implement at least:

### 1. Random Legal

Samples uniformly from the legal action set.

Purpose:

- absolute performance floor;
- stochastic reproducibility test.

### 2. Basic Heuristic

Simple public-information scoring.

May consider:

- known winning captures;
- avoiding known losing captures;
- mobility;
- moving toward opponent territory;
- avoiding obviously useless back-and-forth behavior;
- simple preference for attacks over random quiet moves where sensible.

Keep it intentionally modest.

### 3. Tactical Rule-Based

Stronger local reasoning.

May consider only public/observer-legal facts such as:

- known combat outcomes from revealed ranks;
- Bomb/Flag knowledge when legally revealed;
- Miner value against known Bombs;
- Spy/Marshal special interaction when known;
- multi-square Scout opportunities;
- immediate threats/evasions;
- protecting valuable revealed pieces;
- avoiding obvious tactical sacrifices;
- immediate Flag capture;
- immediate prevention of known Flag capture.

### 4. Strategic Rule-Based

Add longer-horizon public-information heuristics such as:

- revealed material preservation;
- preserving Miners while unresolved Bombs remain;
- Scout information-gathering value;
- pressure on likely/known defensive zones without using privileged type knowledge;
- Flag-defense behavior using only own Flag location and public opponent information;
- mobility/space;
- avoiding unnecessary exposure of high-value known pieces;
- draw/battleless-counter awareness.

This is still a hand-coded baseline, not an attempt at superhuman Stratego.

## Required stress/unusual policies

Implement at least **four** distinct stress policies.

Suggested types:

- Scout-heavy aggression;
- Miner rush/aggressive Bomb hunting;
- defensive/draw-seeking;
- high-attack-frequency / pressure policy;
- low-information-conservation policy;
- deliberately irregular/high-entropy policy.

Stress policies do not need to be stronger than Strategic.

Their purpose is to create different game distributions and expose brittle future policies.

## Policy determinism

For every stochastic policy:

```text
same public input + same policy seed -> same action
```

For deterministic policies:

```text
same public input -> same action
```

Ties must be broken deterministically before optional seeded sampling.

## Hidden-information safety

No baseline may inspect the full game state or true opponent types.

Create policy-level differential tests:

1. generate a valid public situation;
2. clone it;
3. permute true identities among still-hidden opponent pieces while preserving public constraints;
4. build the acting player's legal policy input in both cases;
5. use the same policy seed;
6. require the selected action and any model-facing/public diagnostic score to remain identical.

Agent 4 owns the full 100,000+ audit, but Agent 2 must provide a meaningful local regression suite across all policies.

## Policy legality

For every policy:

- run many random valid positions;
- require selected action is in the legal-action list;
- require no empty legal set is presented for a nonterminal state;
- retain engine-side legality rejection.

## Diagnostics

Policies may emit diagnostics such as:

```text
top candidate actions
heuristic score components
chosen rule family
```

Diagnostics must not contain hidden true opponent types.

Keep diagnostics deterministic and serializable.

## Data file

Write:

```text
reports/phase_4_data/agent_02_baseline_agents.json
```

Minimum:

```text
agent
status
policy_ids
policy_versions
stress_policy_ids
positions_tested_per_policy
illegal_actions
determinism_trials
determinism_failures
local_hidden_permutation_trials
local_hidden_permutation_failures
behavior_summary
test_total
test_passed
test_failed
files_created
files_modified
```

## Report section

Append only:

```markdown
## 2. Agent 2 — Baseline and Stress Opponents
```

## Completion gate

PASS only if:

- Random, Basic, Tactical, and Strategic policies exist;
- at least four stress policies exist;
- every policy uses Agent 1's observer-safe interface;
- no policy returns illegal actions in acceptance testing;
- deterministic/stochastic reproducibility tests pass;
- local hidden-information differential tests pass with zero unexplained differences;
- earlier tests remain green.

Do not tune policies to arbitrary target win percentages yet. Agent 4 owns calibration.
