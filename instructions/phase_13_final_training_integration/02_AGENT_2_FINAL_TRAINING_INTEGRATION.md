# Phase 13 — Agent 2
## Exact Final-Training Integration

## Mission

Build the exact Phase 14 final-training runner from the frozen Agent 1 contract using already-accepted project components.

This agent integrates and tests the machinery. It does **not** tune the model, run the 90-minute rehearsal, perform strength experiments, change the Agent 1 contract, or start the 168-hour Phase 14 run.

## 1. Required starting artifacts

Begin only from an accepted Agent 1 state containing at minimum:

```text
phase13_final_training_contract_v1
phase13_setup_census_v1
phase14_setup_source_v1
phase14_checkpoint_selection_pack_v1
phase14_checkpoint_selection_rule_v1
```

Verify their identities before implementation. Do not reinterpret or silently modify their values.

## 2. Preserve project history

Do not modify accepted artifacts from Phase 9, Phase 10, Phase 11, Phase 11B, Phase 12, or Phase 13 Agent 1 evidence.

All Phase 13 Agent 2 implementation must be additive or explicitly scoped to the new Phase 13/14 runner.

## 3. Exact training stack

Integrate:

```text
accepted Phase 9 C1 starting checkpoint
        ↓
accepted direct-policy RL learner
        ↓
Phase 14 setup source
        ↓
population self-play
        ↓
current neural opponent
bounded active historical pool
handcrafted diversity opponents
        ↓
accepted trajectory reconstruction
        ↓
policy/value/belief-auxiliary updates
        ↓
hot checkpoints
historical archive
candidate checkpoints
candidate evaluator
telemetry
absolute deadline controller
```

Search is not part of this graph.

## 4. Accepted Phase 9 objectives

Implement the exact objective/optimizer values frozen by Agent 1. If Agent 1 found that Phase 9 contains a belief auxiliary objective, preserve it exactly as contracted.

Do not substitute Agent 1C training loss. Do not add search-derived supervision. Do not alter objective weights.

## 5. Conservative LR schedule

Implement exactly the main and late LRs frozen by Agent 1.

Requirements:

```text
main LR active from original run start
late LR activates at the exact frozen wall-clock transition
```

The transition is tied to current time minus original run start, not optimizer step, successful compute hours, or time since latest restart. Downtime does not postpone it.

No dynamic LR edits.

## 6. Main / late population schedule

Implement exactly the frozen main and late opponent mixtures.

The population mixer must support:

```text
current neural policy
historical neural opponent
Strategic
Tactical
Scout-rush
Miner-rush
Information-miser
```

The current/historical ratio changes only at the frozen late transition. The handcrafted share remains exactly as frozen. No adaptive reweighting from results.

## 7. Historical archive

Create durable archive snapshots every 2 hours of original Phase 14 elapsed wall-clock.

The archive contains every durable Phase 14 snapshot. Archive identity/order must remain deterministic across restart. Do not prune archive entries merely because they are not active self-play opponents.

## 8. Active historical pool

Implement the exact bounded-pool algorithm frozen by Agent 1.

Requirements:

```text
permanent anchors preserved
older/middle/recent membership deterministic
pool bounded to frozen size
no tournament admission
sampling weights exactly frozen
early-run missing-category behavior deterministic
```

Hot checkpoints must save enough state that after restart:

```text
active pool membership before crash
==
active pool membership after resume
```

for the same logical point in the run.

Do not reconstruct a new pool from wall-clock alone if that could differ from the saved logical state.

## 9. Checkpoint system

### Hot resume

Cadence:

```text
15 minutes
```

Persist every field required by the Agent 1 contract. Validate checkpoint writes before rotating older hot checkpoints. Keep at least the most recent four valid hot checkpoints.

### Durable archive

Cadence:

```text
2 hours
```

Write to the frozen durable storage location.

### Candidate checkpoints

Cadence:

```text
6 hours
```

Mark the corresponding durable archive snapshot as a final-policy candidate. Support hour 0 through hour 168. The final checkpoint is only a candidate.

## 10. Candidate evaluation pipeline

Use the exact frozen candidate pack and selection-rule metadata.

The candidate evaluator must:

- use direct C1 only;
- use the same fixed 128-game pack for every candidate;
- record per-opponent and overall EWR;
- never change training behavior;
- never stop training early;
- never change LR/population/setup;
- never extend the run.

Candidate evaluation should be operationally decoupled enough that:

```text
evaluation failure != training failure
```

If evaluation cannot run at the scheduled moment, preserve the candidate, record evaluation pending, continue Phase 14, and permit later evaluation on the exact same pack.

Do not use search.

## 11. Absolute deadline controller

Implement:

```text
run_start_utc
run_deadline_utc = run_start_utc + 168 hours
```

Persist them in hot checkpoints. On resume, reuse the persisted deadline. The runner must never produce a new 168-hour duration on restart.

At/after deadline:

- no new collection units;
- no new optimizer step may begin;
- finalize according to the frozen bulk-sync boundary;
- save final state;
- preserve hour-168 candidate;
- close training.

If started after deadline during recovery, finalize immediately without optimizer updates.

## 12. Scheduler / test clock

A real 90-minute rehearsal cannot naturally reach the 2-hour archive cadence, 6-hour candidate cadence, ~75–80% late transition, or 168-hour stop.

Implement a **test-only controllable clock/scheduler seam**. It must be unavailable or disabled in production Phase 14 mode.

Use it in tests to prove:

```text
2h archive trigger
6h candidate trigger
main→late transition
LR transition
historical-pool evolution
168h shutdown
restart with original deadline
```

Production uses the real wall clock. The test seam must not change production semantics.

## 13. Storage integration

Implement the frozen internal hot-checkpoint path, external durable archive path, trajectory/shard retention policy, logging path, and candidate-evaluation path.

Enforce or monitor the Agent 1 reserve requirement.

Do not delete earlier accepted project evidence.

If rolling raw-shard deletion is part of the contract, delete only Phase 14 shards explicitly marked consumed, disposable, and safe_to_delete.

## 14. Telemetry and controls

Expose at minimum:

```text
start UTC
deadline UTC
elapsed
remaining
optimizer step
current LR
current segment
current opponent mixture
active historical pool
games generated
positions generated
throughput
losses
gradient norm
advantage-filter fraction
draw/game-length stats
latest hot checkpoint
latest archive checkpoint
latest candidate checkpoint
candidate evaluation status
disk free/used
worker status
failure counters
```

Emergency stop must remain available. Normal control paths should not expose mutable edits for frozen training parameters.

## 15. Failure handling

Implement recoverable handling for:

```text
worker crash
training process crash
temporary MPS failure
interrupted shard write
reboot/resume
```

A recoverable restart must load the newest valid hot checkpoint, restore optimizer/model/EMA, restore RNG/stream state, restore active historical pool, restore scheduler state, restore original start/deadline, restore shard cursor, and continue only if before deadline.

Detect and stop on unrecoverable integrity failures such as no valid checkpoint, wrong starting model, checkpoint state inconsistency, configuration digest mismatch, or irrecoverable optimizer corruption.

Do not silently start a fresh logical run.

## 16. Short integration tests only

Run short sanity executions sufficient to prove:

```text
model updates occur
losses are finite
belief auxiliary objective runs if contracted
population mixer works
Phase 14 setup source works
hot checkpoint writes/loads
archive checkpoint writes
candidate marking works
candidate evaluator can run
deadline controller works
test-clock late transition works
active historical pool evolves deterministically
resume restores exact logical state
storage/telemetry paths work
search is absent from training
```

Do not run the 90-minute rehearsal. Do not run a strength tournament.

## 17. Identity / configuration binding

Create a deterministic config digest for the fully integrated Phase 14 runner inputs.

Bind at minimum:

```text
starting Phase 9 checkpoint
training objective config
main LR
late LR
transition time
main opponent mixture
late opponent mixture
historical pool algorithm
setup source identity
checkpoint cadences
candidate evaluation pack
selection rule
storage policy
deadline semantics
```

This digest becomes an input to Agent 4's immutable launch manifest.

## 18. Deliverables

Create at minimum:

```text
Phase 14 final-training runner/module(s)
Phase 14 scheduler/deadline module
historical archive/pool module
candidate evaluation module
storage/retention integration
telemetry/control integration
tests for the new Phase 13/14 machinery
phase13_agent_02_report.md
phase13_agent_02_summary.json
phase13_integrated_training_config_v1
```

Record all short-test results.

## 19. Stop condition

Stop when the exact frozen Agent 1 contract is implemented, all short integration tests pass, production search is absent from training, test-clock scheduler checks prove long-horizon events, and no 90-minute rehearsal has begun.

Do not begin Agent 3 automatically.
