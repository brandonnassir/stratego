# Phase 9 — Agent 3
# Self-Play Collector and Crash-Safe Rollout Store

## Mission

Build the production Phase 9 neural/rule self-play collector and the crash-safe per-iteration rollout store.

This is the first agent allowed to run **neural self-play collection**, but it must not perform meaningful PPO/RL optimization.

The collection boundary must make later importance ratios trustworthy.

## Prerequisites

Require Agents 1–2 `PASS` and formal acceptance.

Verify schedule/population digests and exact counts from live source.

Verify accepted Phase 8 checkpoint SHA and C1 identity.

### Mandatory corpus resolver check

Resolve through `synthetic_corpus.default_corpus_root()` and require the accepted Phase 8 corpus digests. Do not hard-code the absolute path into collection or rollout code. A mismatch is `BLOCKED`.

## Behavior snapshot

At collection start for an iteration:

- load/freeze one immutable behavior checkpoint;
- compute and record its file/model identity;
- put model in evaluation/inference mode;
- prohibit optimizer mutation while collection is active;
- every current-policy side in the iteration must use this exact behavior snapshot.

A single logical iteration containing two behavior snapshot identities is an immediate hard failure.

## Neural action selection

For learner/current neural sides:

- construct model input only from the frozen observer-safe observation;
- mask illegal actions;
- compute the frozen temperature-1 legal softmax;
- sample using the deterministic per-game/per-ply neural RNG identity;
- store the realized action;
- store the behavior information exactly as frozen by Agent 1;
- store behavior WDL output required by Agent 4 targets;
- store/check the exact behavior snapshot identity.

Evaluation argmax is not used for training collection.

## Rule/stress/historical actions

Historical neural sides use their own immutable checkpoint identity and the Agent 1-frozen historical behavior rule.

Fixed rule/stress sides use their accepted Phase 4 decision path.

Opponent decisions do not become learner training examples unless `learner_control` includes that side.

## Rollout store

Implement:

```text
phase9_rollout_store_v1
```

Prefer reuse of proven Phase 6/8 shard/commit machinery where semantically valid.

Each game becomes visible/trainable only after:

```text
trajectory payload
+
Phase 9 metadata
+
commit record
```

all verify.

Required metadata includes:

```text
phase9_game_id
rl_iteration
bucket
red/blue policy identities
learner_control
behavior snapshot identity
historical identity if any
setup provenance
terminal result/reason
game length
Phase 9 contract versions
```

Do not mutate `trajectory_v1` semantics in place.

## Iteration state machine

Implement and persist:

```text
COLLECTING
SEALED
TRAINING
EVALUATED
COMMITTED
```

Agent 3 owns `COLLECTING -> SEALED`.

A rollout may be sealed only when:

- exact scheduled game count is present;
- all schedule IDs match;
- no duplicates;
- no unscheduled games;
- no orphan payload/metadata records;
- every committed payload decodes;
- every game replays legally;
- every current neural decision has verifiable behavior information;
- one behavior identity only;
- setup provenance reconstructs.

After `SEALED`, payload bytes are immutable.

## Crash/recovery

Inject failures at all critical points:

```text
before payload
after payload
after metadata
during commit
at shard rollover
between games
process kill
```

Recovery must:

- expose only committed games;
- discard/truncate uncommitted tails;
- never regenerate an already committed logical game;
- resume from `scheduled - committed`;
- converge to the same sealed rollout digest as a clean run.

Worker count may change across resume without changing logical bytes/digests except where file layout is explicitly excluded from identity.

## Behavior reproduction audit

Before handoff, independently reproduce the stored behavior quantity from the frozen checkpoint on at least:

```text
100,000 learner-controlled neural decisions
```

Require Agent 1's numeric tolerance.

Audit:

- acting player;
- observation digest;
- legal set;
- action frame;
- behavior distribution/log-prob;
- sampled action legality;
- WDL output;
- behavior snapshot identity.

Zero mismatches required.

## Collection soak

Run a substantial real C1 collection soak using the frozen Phase 8 anchor as the behavior snapshot.

Minimum:

```text
>= 8,192 complete games
population schedule semantics active
all four population buckets represented where possible
```

This is infrastructure collection only. Do not optimize from it.

Measure:

```text
positions/s
games/s
CPU utilization
MPS utilization
peak RSS
MPS memory
bytes/game
bytes/position
compression ratio
storage/hour projection
data integrity
```

If collector topology differs from Phase 6 production, justify using measurement only; do not alter logical game identity.

## Observer-safety boundary

The model input may contain only observation tensors and legality required for action masking.

Privileged truth may exist in rollout metadata for belief labeling later, but must be inaccessible from model input.

Add positive-control tests proving the boundary audit detects planted privileged information.

## Artifacts

Create:

```text
reports/phase_9_data/agent_03_rollout_store.json
reports/phase_9_data/agent_03_collection_soak.json
reports/phase_9_data/agent_03_behavior_reproduction.json
```

## Completion gates

Require:

```text
agents1_2_pass
corpus_resolver_verified
corpus_digests_match
behavior_snapshot_immutable
one_behavior_identity_per_iteration
neural_actions_legal
behavior_storage_matches_contract
behavior_reproduction_ge_100k
behavior_reproduction_mismatches_zero
rollout_commit_protocol_pass
crash_resume_converges
orphan_records_zero
duplicate_game_ids_zero
unscheduled_games_zero
replay_illegal_actions_zero
setup_provenance_mismatches_zero
observer_input_leaks_zero
collection_soak_ge_8192_games
no_rl_optimizer_steps
full_suite_green
```

## Forbidden

Do not:

- compute/optimize PPO;
- tune population proportions;
- tune temperature;
- change behavior storage after seeing results;
- train on the soak;
- open Phase 9 final test;
- modify Phase 8 checkpoint;
- begin Agent 4 target design beyond implementing Agent 1's frozen fields.

## Handoff to Agent 4

Provide:

- sealed rollout reader;
- random-access reconstruction;
- behavior quantity access;
- behavior WDL outputs;
- learner-control masks;
- privileged target-only state access;
- rollout/behavior digests;
- crash-safe iteration state;
- independent reproduction evidence.

Agent 4 must independently reconstruct RL targets rather than trust collector bookkeeping.
