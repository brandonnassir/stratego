# Phase 3 Agent 3 — Compact Trajectory Storage, Snapshots, and Reconstruction

## Role and scope

You are **Agent 3** in a sequential Phase 3 implementation. Work only on the task in this file.

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

## Prerequisite

Agent 2 must be `PASS`.

Read Sections 1-2 of the shared report and:

```text
reports/phase_3_data/agent_01_batch_equivalence.json
reports/phase_3_data/agent_02_shared_memory_scaling.json
```

If prerequisites are not PASS, append Section 3 as `BLOCKED` and stop.

## Objective

Implement the compact self-play trajectory/reconstruction layer.

This agent does **not** implement MPS inference or the final coordinator.

## Files you own

Create/adapt:

```text
stratego/training/trajectory.py
stratego/training/reconstruction.py
tests/training/test_trajectory.py
tests/training/test_reconstruction.py
scripts/run_phase3_agent03.py
reports/phase_3_data/agent_03_trajectory_reconstruction.json
```

You may add serializer/helper modules under `stratego/training/` if clearly needed.

## Storage principle

Do not store the full `127 x 10 x 10` observation tensor per move.

Store compact facts and reconstruct through the frozen reference engine.

## Required game record

At minimum:

```text
game_id
environment_id
generation
rules_version
observation_version
red_setup
blue_setup
first_player
setup_family/setup_id when available
terminal_result
terminal_reason
final_ply
collection policy/checkpoint identifiers when available
```

## Required decision record

At minimum:

```text
game_id
ply
acting_player
selected_action_id
legal_action_ids
old_probabilities_over_legal_actions
win_draw_loss_prediction
collection_policy_version
snapshot_reference
```

For this agent, if there is not yet a real model, use a deterministic synthetic legal-action probability distribution and deterministic synthetic three-class values to verify storage fidelity. The schema must later accept real Agent 4/5 outputs unchanged.

Never store a dense 10,000-probability vector in the trajectory record when the legal set is small.

## Snapshot strategy

Support configurable snapshot intervals:

```text
16
32
64
```

Initial default is 32 plies.

Use compact reference-engine snapshots plus action deltas.

A historical position at ply `p` must reconstruct from:

```text
game record
+ nearest snapshot at or before p
+ subsequent actions
-> frozen reference state
-> observation_v2_1_127ch
-> legal actions/mask
-> privileged belief target
```

The belief target must remain a training target and must not enter the reconstructed policy observation.

## Exact reconstruction validation

Generate trajectory data from live play, then reconstruct sampled historical decisions.

Acceptance target:

- at least **1,000,000 reconstructed historical decisions** total;
- zero unexplained mismatches in:
  - full state fingerprint where available;
  - acting player;
  - `observation_v2_1_127ch`;
  - legal-action list;
  - legal mask on sampled checks;
  - public knowledge;
  - privileged belief target;
  - stored selected action;
  - environment/generation/game identity.

Use deterministic seeds.

If 1,000,000 full dense-mask comparisons would dominate runtime/storage, it is acceptable to compare dense masks on a large stratified subset while still comparing legal-action lists for all 1,000,000. Record the exact counts.

## Snapshot interval benchmark

For 16, 32, and 64:

Measure:

- average serialized bytes/game;
- median bytes/game;
- snapshot bytes/game;
- decision-record bytes/game;
- reconstruction positions/second;
- mean replayed actions per reconstruction;
- p95 replayed actions per reconstruction.

Choose a recommended interval based on measured storage/reconstruction tradeoff.

Do not automatically choose the fastest interval if its storage cost is disproportionate. Explain the tradeoff.

## Sparse decision-storage checks

Verify:

- legal action IDs are unique and sorted or deterministically ordered;
- old probabilities match legal IDs one-for-one;
- probabilities are finite, nonnegative, and sum to approximately 1;
- selected action appears in legal IDs;
- collection policy version is preserved;
- three-class value prediction has three finite entries and is normalized if stored as probabilities.

## Data files

Primary:

```text
reports/phase_3_data/agent_03_trajectory_reconstruction.json
```

Optional raw:

```text
reports/phase_3_data/agent_03_snapshot_interval_raw.csv
```

JSON minimum:

```text
agent
status
games_generated
decisions_stored
historical_decisions_reconstructed
state_mismatches
observation_mismatches
legal_list_mismatches
legal_mask_checks
legal_mask_mismatches
belief_target_mismatches
identity_generation_mismatches
snapshot_intervals_tested
snapshot_interval_results
recommended_snapshot_interval
mean_replay_bytes
mean_decision_bytes
estimated_million_game_bytes
reconstruction_positions_per_second
files_created
files_modified
```

## Report section

Append only:

```markdown
## 3. Agent 3 — Trajectory Storage and Reconstruction
```

## Completion gate

PASS only if:

- trajectory schema is compact and versioned;
- no full observation tensor is stored per decision;
- at least 1,000,000 historical decisions reconstruct with zero unexplained mismatch in required fields;
- all snapshot intervals are measured;
- sparse legal-probability records validate;
- belief targets remain separate from policy inputs;
- a recommended snapshot interval is documented.

Do not implement the neural network.
