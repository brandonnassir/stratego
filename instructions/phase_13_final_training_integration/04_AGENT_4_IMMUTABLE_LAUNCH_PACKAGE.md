# Phase 13 — Agent 4
## Immutable Phase 14 Launch Package and GO / NO-GO Decision

## Mission

Review the completed Phase 13 contract, integration, setup census, and rehearsal; verify all launch-readiness gates; freeze one immutable Phase 14 launch package; and return either:

```text
GO
```

or:

```text
NO-GO
```

Do **not** start the 168-hour run.

This agent is a launch-readiness reviewer and packager, not another experimenter.

## 1. Required inputs

Review the accepted outputs from:

```text
Phase 13 Agent 1 — Final Training Contract + Setup Census
Phase 13 Agent 2 — Exact Final-Training Integration
Phase 13 Agent 3 — 90-Minute Crash/Resume Rehearsal
```

Verify their identities and sequential project state. Also verify preserved upstream identities required by the contract.

## 2. Do not reopen closed decisions

Do not retune or reconsider merely because a different choice seems attractive now.

Do not change:

```text
starting Phase 9 checkpoint
main LR
late LR
main/late transition
main opponent mixture
late opponent mixture
belief auxiliary objective
historical archive cadence
active historical pool algorithm
historical sampling weights
setup source
15-minute checkpoint cadence
2-hour archive cadence
6-hour candidate cadence
candidate evaluation pack
post-run checkpoint-selection rule
168-hour deadline semantics
storage retention policy
```

If a frozen value is internally inconsistent or implementation-invalid, return `NO-GO`. Do not silently fix strategic choices at Agent 4.

## 3. Verify search exclusion

Confirm structurally that Phase 14 training does not invoke:

```text
Phase 12 TINY
Phase 12 SMALL
Phase 12 MEDIUM
oracle search
Agent 1C as move policy
```

The production training move model must be the direct continuing C1 policy. Search is rebound only after final-policy selection and later belief specialization.

## 4. Verify starting model roles

Freeze:

```text
Phase 14 starting policy/value:
    accepted Phase 9 C1

Phase 14 belief auxiliary:
    accepted Phase 9 learner behavior, if present

Agent 1C:
    preserved engineering artifact
    not used as Phase 14 policy/value
```

Bind exact checkpoint/state digests.

## 5. Verify conservative continuation schedule

Confirm:

```text
main continuation LR < accepted Phase 9 LR
late continuation LR < main continuation LR
```

Confirm exact multipliers/values match Agent 1's accepted contract. Confirm the late transition is tied to original wall-clock time and survives restart. No live LR tuning may be enabled in the normal Phase 14 control path.

## 6. Verify opponent population contract

Confirm total handcrafted/rule/unusual share is within the frozen 10–15% engineering target.

Confirm representation from:

```text
Strategic
Tactical
Scout-rush
Miner-rush
Information-miser
```

Confirm the late segment shifts neural weight from current toward historical without changing rules mid-run. No adaptive outcome-based population tuning.

## 7. Verify historical archive and active pool

### Archive

Confirm:

```text
durable snapshot every 2 hours
all durable snapshots retained per storage policy
```

### Active pool

Confirm:

```text
bounded size
permanent Phase 8 anchor
permanent accepted Phase 9 anchor
deterministic older/middle/recent Phase 14 membership
fixed historical sampling weights
no tournament admission
```

Confirm hot checkpoints preserve exact pool membership/state. Confirm Agent 3 recovery evidence shows pool continuity.

## 8. Verify setup census / final setup source

Review:

```text
phase13_setup_census_alarm_policy_v1
phase13_setup_census_v1
phase14_setup_source_v1
```

Confirm the alarm policy was frozen before sampling.

Confirm the census reports:

```text
Flag row distribution
immediate Scout exposure
pre-play/trivial capture patterns
source branch
reflection effect
perturbation effect
```

If a repair occurred:

- confirm it is narrowly scoped;
- confirm Phase 10 evidence was not rewritten;
- confirm the corrected Phase 14 setup source has a new identity;
- confirm minimum verification was rerun.

If a required defect/pathology remains unresolved, return `NO-GO`.

## 9. Verify checkpoint hierarchy

Confirm:

```text
hot resume       every 15 minutes
durable archive  every 2 hours
policy candidate every 6 hours
```

Confirm hour 0 and hour 168 candidate handling. Confirm hour 168 is **not automatically selected**.

## 10. Verify fixed candidate evaluation / selection

Confirm the candidate pack is frozen and identical for every candidate. Confirm direct-policy evaluation only. Confirm no search.

Confirm the evaluator cannot:

```text
stop training
change LR
change opponent mixture
change setup source
change historical pool
extend deadline
```

Confirm the frozen post-run selection rule is exactly implemented.

The launch package must state plainly:

> Phase 14 runs for the full 168-hour deadline regardless of intermediate candidate scores. The deployed direct-policy checkpoint is selected only after training ends using the predeclared fixed rule.

## 11. Verify 168-hour deadline contract

The launch package must bind:

```text
duration = 168 hours
deadline = actual launch UTC + 168 hours
```

The actual absolute UTC deadline cannot be known until Phase 14 launches. Freeze the derivation rule, then require the launch script to materialize `run_start_utc` and `run_deadline_utc` once at launch and persist them.

Confirm:

```text
downtime counts against deadline
restart reuses deadline
late transition reuses original start
post-deadline recovery cannot train
```

## 12. Verify rehearsal

Review Agent 3 evidence.

At minimum require successful evidence for:

```text
normal training updates
finite losses
parameter changes
forced process termination
resume from valid hot checkpoint
optimizer state preservation
deadline preservation
active historical pool preservation
worker failure recovery
storage behavior
automatic rehearsal deadline stop
post-deadline recovery refusal
```

If a central recovery guarantee failed and remains unresolved, return `NO-GO`.

## 13. Verify storage readiness

Review actual available external storage and measured rehearsal growth.

Require:

```text
projected Phase 14 usage
+ frozen retention policy
+ 20% safety reserve
```

to fit the intended volume.

Never solve storage pressure by deleting accepted historical project evidence.

If the projection does not fit and no safe frozen retention policy exists, return `NO-GO`.

## 14. Verify monitoring and control

Require status visibility for:

```text
start/deadline
elapsed/remaining
optimizer step
current LR
current segment
population mixture
active historical pool
throughput
losses
checkpoint age
candidate evaluation status
disk usage
worker health
failure counters
```

Require emergency stop and restart/resume procedures. Normal controls must not expose easy mutation of frozen training parameters.

## 15. Launch readiness gates

Evaluate all gates.

### Gate A — Upstream identity

All required accepted starting artifacts match expected identities.

### Gate B — Final training contract

LR, objectives, schedule, mixtures, historical pool, checkpoint rules, selection rule, and deadline semantics are exact and frozen.

### Gate C — Setup safety

Census complete and final setup source frozen; no unresolved defect/pathology blocker.

### Gate D — Training correctness

Integrated runner produces finite legal updates and expected parameter changes; belief auxiliary objective works if contracted.

### Gate E — Recovery

Process crash and worker failure recover without resetting logical training state.

### Gate F — Wall-clock semantics

Original start/deadline and late transition survive downtime/restart; post-deadline recovery performs zero optimizer steps.

### Gate G — Historical system

Archive and bounded active pool evolve deterministically and resume exactly.

### Gate H — Candidate system

6-hour candidate scheduling, fixed evaluation pack, and post-run selection rule function without influencing training.

### Gate I — Storage

Projected 168-hour storage fits frozen policy with reserve.

### Gate J — Controls

Monitoring, emergency stop, and resume function; frozen parameters cannot casually change.

All gates must pass for `GO`. No weighted score. No scientific confidence interval.

## 16. Freeze `phase14_final_training_config_v1`

Create an immutable configuration artifact binding at minimum:

```text
starting Phase 9 checkpoint identity
optimizer family
policy/value/belief auxiliary objectives
main LR
late LR
main/late transition elapsed time
main opponent mixture
late opponent mixture
historical archive cadence
active historical pool algorithm
historical sampling weights
Phase 14 setup source identity
hot checkpoint cadence
durable archive cadence
candidate checkpoint cadence
candidate evaluation pack identity
candidate selection rule identity
storage/retention policy
monitoring config
recovery semantics
deadline derivation rule
search excluded = true
```

Include a deterministic config digest.

## 17. Freeze `phase14_launch_manifest_v1`

Create a launch manifest that binds:

```text
code revision
all upstream identities
phase14_final_training_config_v1 digest
external storage location
internal hot-checkpoint location
launch script identity
resume script identity
candidate evaluator identity
emergency-stop procedure identity
```

The actual launch operation will materialize `run_start_utc` and `run_deadline_utc` exactly once.

## 18. Create `PHASE_14_RUNBOOK.md`

The runbook must be operational and concise.

### Pre-launch

```text
verify Mac power/sleep settings
verify external volume mounted
verify free space
verify clean/frozen code revision
verify checkpoint identities
verify no existing Phase 14 run identity
verify monitoring available
```

### Launch

Provide the exact procedure to start the 168-hour run.

### Normal monitoring

State what to inspect without changing training.

### Recoverable crash

Provide exact resume procedure. State explicitly:

```text
never create a new 168-hour deadline
```

### Worker failure

Document expected recovery behavior.

### Storage warning

Document only actions allowed under the frozen retention policy.

### Emergency stop

Document exact procedure and consequences.

### Deadline

Document expected automatic shutdown behavior.

### Post-run

```text
training closed
hour-168 candidate preserved
complete any missing fixed-pack candidate evaluations
apply frozen checkpoint-selection rule
select final direct-policy checkpoint
do not train further
```

### After selection

State the next project sequence only:

```text
copy selected final C1
dedicated Agent-1C-style belief specialization
freeze final belief provider
rebind Phase 12 search
final machine evaluation
final human evaluation
```

Do not perform those steps in Agent 4.

## 19. Final recommendation

Return exactly one engineering recommendation:

```text
GO
```

if all Gates A–J pass.

Otherwise:

```text
NO-GO
```

with the specific blocking gates and required repairs.

Do not weaken a failed launch-readiness gate because the 168-hour run is inconvenient to delay.

## 20. Deliverables

Create:

```text
phase14_final_training_config_v1
phase14_launch_manifest_v1
PHASE_14_RUNBOOK.md
phase13_agent_04_report.md
phase13_agent_04_summary.json
```

Bind all digests.

## 21. Stop condition

Stop after the immutable launch package and GO/NO-GO recommendation are complete.

Do not start Phase 14. Do not run another rehearsal unless a blocker requires a narrowly scoped repair and review.
