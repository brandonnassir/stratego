# Phase 13 — Agent 3
## 90-Minute Crash/Resume Rehearsal

## Mission

Run one deliberate 90-minute end-to-end rehearsal of the exact Phase 14 training system and prove that it can survive realistic failures without changing the logical run, schedule, population state, or deadline.

This is a reliability rehearsal. It is **not** a strength experiment, learning-rate experiment, opponent-mixture experiment, search experiment, or a reason to change the model based on short-term losses or win rates.

## 1. Prerequisites

Begin only from an accepted Agent 2 implementation.

Verify:

```text
Agent 1 contract identity
Agent 2 integrated config identity
starting Phase 9 checkpoint identity
Phase 14 setup source identity
candidate pack identity
checkpoint-selection rule identity
```

Do not modify the frozen training strategy.

## 2. Rehearsal duration

Create a rehearsal run with an absolute:

```text
90-minute wall-clock deadline
```

Use the same deadline semantics as Phase 14:

```text
rehearsal_start_utc
rehearsal_deadline_utc = start + 90 minutes
```

Downtime during rehearsal counts against the 90 minutes. A restart must reuse the original rehearsal deadline. Do not add downtime back.

## 3. Use the final training configuration

Use the exact Phase 14 configuration except where the real 168-hour duration is replaced by the 90-minute rehearsal deadline.

Do not scale up LR because the rehearsal is shorter. Do not alter opponent weights to make events happen faster. Do not use search.

Use the actual Phase 9 start, Phase 14 setup source, main continuation LR, main population mixture, belief auxiliary objective, hot-checkpoint machinery, storage/retention machinery, and telemetry.

The real late-phase transition should **not** be moved earlier in the actual training logic. Long-horizon scheduler behavior is verified separately through the Agent 2 test-clock seam.

## 4. Planned failure sequence

### Segment A — normal operation

Approximately:

```text
0–30 minutes
```

Run normally.

Verify:

```text
optimizer steps increase
model parameters change
losses remain finite
workers generate games
hot checkpoints appear
telemetry updates
```

### Failure 1 — full process termination

At approximately 30 minutes:

- force termination of the main training process;
- simulate a real process-level failure rather than a graceful save;
- do not manually create a special checkpoint immediately beforehand.

Record the time of failure, last completed optimizer step, latest valid hot checkpoint, active historical pool, RNG/stream state identity, and deadline.

Restart through the production recovery path.

### Segment B — resumed operation

Verify after resume:

```text
original deadline unchanged
optimizer step not reset
model/optimizer restored
EMA restored
RNG/stream state restored
population schedule restored
active historical pool restored
storage/shard cursor restored
no fresh logical run created
```

Continue training.

### Failure 2 — worker failure

Later in the rehearsal:

- kill or force failure of one CPU worker;
- allow the normal worker-recovery path to handle it.

Verify:

```text
main learner survives
replacement/recovery behaves as designed
no optimizer reset
no population reset
no deadline reset
no corrupted trajectory accepted
```

### Segment C — natural deadline

Continue until the original 90-minute deadline. The run must stop automatically at that deadline despite earlier downtime.

## 5. Resume boundary checks

Demonstrate that recovery does not produce:

```text
duplicate optimizer work
skipped logical optimizer state
fresh RNG initialization
different active historical pool
reset archive cursor
reset candidate scheduler
reset main/late scheduler
reset start time
reset deadline
```

Where exact bitwise continuity is practical, verify it. Where asynchronous worker timing prevents bitwise comparison, verify logical counters/state and explain the limitation. Do not weaken the requirement silently.

## 6. Hot checkpoint verification

The 15-minute hot-checkpoint system must operate throughout rehearsal.

Verify:

```text
checkpoints created at intended cadence
newest valid checkpoint selected on recovery
corrupt/incomplete checkpoint not selected
recent valid checkpoints retained
checkpoint includes required scheduler/pool/deadline state
```

Test checkpoint readability after rehearsal completes.

## 7. Storage verification

Monitor actual growth during the rehearsal.

Report:

```text
raw shard GiB/hour
checkpoint growth
log growth
external-write behavior
free-space change
projected 168-hour usage
```

Compare the measured rate with Agent 1's storage projection. If the measured rate materially threatens the frozen reserve, classify it as a launch blocker.

Do not delete earlier accepted evidence.

## 8. Candidate checkpoint / evaluator plumbing

The 90-minute run will not naturally reach the real 6-hour candidate cadence.

Do **not** change the production cadence.

Use only the Agent 2 test-clock/scheduler seam outside the production rehearsal to verify:

```text
6-hour candidate event fires
candidate is marked from archive
fixed evaluation pack launches/queues
evaluation failure does not affect training
evaluation result cannot change training config
hour-168 candidate logic exists
```

Do not perform a new candidate-strength experiment.

## 9. Late-phase transition plumbing

The rehearsal does not naturally reach the real ~75–80% Phase 14 transition. Do not compress the production schedule.

Using the test-clock seam, verify separately:

```text
main→late opponent mixture transition
main→late LR transition
transition tied to original wall-clock
downtime does not postpone transition
restart preserves which segment should be active
```

## 10. Deadline tests

Verify two cases.

### Normal rehearsal deadline

At original +90 minutes:

```text
no new collection launched
no optimizer step begun after deadline
final state written
run marked complete
```

### Recovery started after deadline

Using a controlled test:

- load a valid checkpoint;
- set current test time after persisted deadline;
- invoke recovery.

Required behavior:

```text
0 optimizer steps
immediate finalization
deadline not extended
```

## 11. Monitoring / controls

Verify live status accurately reports:

```text
elapsed
remaining
deadline
optimizer step
current LR
current schedule segment
opponent mixture
active historical pool
worker status
checkpoint age
disk usage
losses
failure/recovery events
```

Emergency stop must function. Do not expose mutable training controls that violate the freeze.

## 12. Rehearsal result must not change training strategy

Do not use rehearsal EWR, short-term policy loss, short-term value loss, short-term belief loss, or short-term game outcomes to alter LR, loss weights, opponent mixture, historical pool algorithm, setup source, candidate selection rule, or checkpoint cadence.

The only allowed post-rehearsal changes are implementation/reliability defect fixes. If a defect is fixed, rerun only the minimum affected verification unless the fix materially changes the end-to-end recovery path.

## 13. Readiness checks

Report at least:

```text
training updates finite
parameters change
belief auxiliary objective functioning if contracted
forced process crash recovered
optimizer state preserved
original rehearsal deadline preserved
active historical pool preserved
worker failure recovered
storage remained safe
hot checkpoints readable
test-clock 2h archive event works
test-clock 6h candidate event works
test-clock late transition works
test-clock 168h shutdown works
post-deadline recovery refuses training
search absent from training
```

## 14. Deliverables

Create:

```text
phase13_rehearsal_v1
phase13_agent_03_report.md
phase13_agent_03_summary.json
rehearsal logs
recovery evidence
checkpoint/resume evidence
storage projection update
scheduler/test-clock evidence
```

Explicitly list any defect encountered and any narrow fix applied.

## 15. Stop condition

Stop after the single 90-minute rehearsal completes, required crash/resume checks pass or blockers are documented, worker failure check completes, long-horizon scheduler tests complete, storage projection is updated, and all readiness evidence is written.

Do not start Phase 14. Do not begin Agent 4 automatically.
