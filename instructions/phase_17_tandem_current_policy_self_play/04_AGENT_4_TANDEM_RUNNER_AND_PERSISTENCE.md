# Phase 17 — Agent 4
## Tandem move/setup runner, persistence, schedules, telemetry, and guards

## Mission

Integrate the accepted Agent 2 move learner and Agent 3 setup learner into one bounded,
restartable Phase 17 system. Freeze the 12-hour schedules, implement paired raw/EMA
checkpointing and export, preserve active games exactly, and expose sufficient
telemetry for collapse detection.

You do not redesign either model, run the external MacBook setup, authorize launch,
or run the 12-hour production job. Read
`00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` completely and require verified Agent
1–3 handoffs with both implementation readiness fields true.

## 1. Ownership and integration discipline

Own:

```text
stratego/training/phase17/runner.py
stratego/training/phase17/checkpoint.py
stratego/training/phase17/queue.py
stratego/training/phase17/telemetry.py
stratego/training/phase17/supervisor.py
stratego/training/phase17/export.py
tests/training/phase17/test_runner_*.py
tests/training/phase17/test_checkpoint_*.py
tests/training/phase17/test_supervisor_*.py
scripts/run_phase17_preflight.py
scripts/run_phase17_training.py
reports/phase17/agent_04_*
```

Do not alter Agent 2/3 behavior to make integration convenient. If their handoff APIs
or the shared schema disagree, stop and reconcile through a versioned amendment with
the owning agent; no adapter may silently change identity, targets, or perspective.

## 2. Iteration order

Implement one explicit bulk-synchronous iteration:

1. bind the current raw move and setup snapshot identities;
2. refill fresh setup pools for that setup snapshot;
3. advance the persistent population until exactly the move-transition budget lands;
4. create replacement games immediately, drawing both setups from the bound raw setup
   distribution and attaching both behavior episodes;
5. enqueue both setup episodes when a game finishes;
6. construct boundary-bootstrapped move targets and update the move model for one epoch;
7. consume exactly the frozen setup-sequence budget when enough eligible completed
   episodes are available and update the setup model for five epochs;
8. update independent KL controllers and raw-to-EMA states;
9. atomically checkpoint and emit the telemetry row;
10. rebind every active game to the newly updated raw move snapshot for its next move.

A setup update may be explicitly skipped for insufficient eligible episodes. Do not
reuse episodes, shrink an update silently, or block the move iteration indefinitely.
Warm-up and starvation status are telemetry fields.

## 3. Sustainable fixed setup budget

Use Agent 3's throughput and a bounded local rehearsal to select a fixed number of
setup side-episodes per setup update. The budget must:

- keep five setup epochs affordable;
- be sustainable under the lower observed game-completion rate;
- consume each episode exactly once;
- impose a maximum accepted policy age frozen by Agent 1;
- retain enough queue reserve to avoid alternating huge and empty updates.

Freeze queue capacity, update batch, warm-up minimum, max age, and backlog/age alarms
in the production config. Overflow or age rejection is an explicit counted event and
must never prefer short games invisibly. If no unbiased bounded policy is possible,
stop for operator review.

## 4. Schedule-horizon rehearsal

Run only the minimum bounded rehearsal needed to measure steady-state tandem
iteration time. It must include actual move forward passes, boundary target creation,
one move epoch, setup generation, and five setup epochs—not a mocked timing path.

Estimate the expected number of iterations `N` in 12 active training hours and freeze:

- `N` and the measurement rows used;
- `n_ref = ceil(0.125 * N)`;
- the complete move LR/entropy curve;
- the complete Agent 1 setup LR/entropy curve;
- both epoch counts and EMA decays.

Do not tune strength from the rehearsal. If observed production speed differs, retain
the frozen schedule horizon; telemetry records the difference.

## 5. Exact joint persistence

Implement an atomic paired checkpoint. It must either load completely and verify or
refuse completely. Include every field in common-contract section 10 plus:

- monotonic checkpoint generation and parent identity;
- active-training elapsed time and next 30-minute export boundary;
- current window partial-transition state, if hot checkpoints can occur mid-window;
- setup episodes attached to active games;
- queue membership/consumption markers;
- current setup-pool contents and behavior identity, or an exact documented rule for
  discarding only unused pool candidates on resume;
- telemetry append position and last durable record digest.

Persist active engine states and boundary carry exactly. A planned round trip must
reproduce the same next sampled move, next generated setup, next transition targets,
and next optimizer update from the same checkpoint.

If the engine cannot be serialized faithfully, stop and report the blocker. Do not
inherit Phase 16's behavior of dropping in-flight games on resume.

Use write-to-temporary, fsync where appropriate, digest verification, and atomic rename.
Never overwrite an accepted checkpoint or expose a partial file under its final name.

## 6. Paired evaluation exports

Export immutable candidates containing:

- move EMA state and state digest;
- setup EMA state and state digest;
- architecture/observation/action/rules identities;
- parent joint-checkpoint identity;
- run/config/source identity;
- active-training elapsed time and nominal cadence;
- whether each benchmark lane consumes each model;
- expected file hashes and manifest digest.

Hour 0 export occurs before the first optimizer update. Subsequent export scheduling
uses active-training elapsed time, every 30 minutes through hour 12. Export creation
must not mutate training RNG, raw weights, EMA state, or timing counters.

Agent 5 may wrap this portable bundle for the chosen transport but may not change its
semantic identities.

## 7. Telemetry

Append durable JSONL rows with a schema frozen by Agent 1. Per iteration include:

### Move

- harvested/trained transition counts and boundary count;
- active/completed games, plies, length distribution, terminal reasons;
- move loss components, entropy, KL, beta, clip fraction, grad norm, LR;
- raw/EMA model-state digests and optimizer steps;
- participant/digest ledger for both seats;
- collection, target, optimization, checkpoint, and total wall time.

### Setup

- generated/refilled/unused counts and snapshot identity;
- legality/orientation failures and fallback attempts;
- queue depth, enqueue/consume/reject counts, age distribution, starvation state;
- setup loss components, empirical/predicted entropy, KL, beta, grad norm, LR;
- reflection-class diversity, flag/bomb effective support, concentration;
- raw/EMA state digests and optimizer steps;
- generation and five-epoch optimization time.

### System

- active-training elapsed time, iteration/cadence index, memory high-water marks;
- joint checkpoint/export identities;
- warnings, stop predicates, and external-result status known to the supervisor.

## 8. Collapse supervisor

Implement every immediate and persistent rule from common-contract section 13 using
Agent 1/3 frozen thresholds. Each predicate has a stable code, evidence payload,
consecutive-count state, severity, and reset rule.

The supervisor may safely checkpoint and stop. It may not change LR, KL targets,
entropy coefficients, population size, epoch counts, setup batch, or benchmark cases.
One injected-event test must prove the stop record and safe exit path. Unit tests cover
predicate arithmetic; no broad failure-injection campaign is required.

## 9. Integration verification

Keep the integration rehearsal bounded:

- at least one full transition iteration with completed and unfinished games;
- at least one real setup update from completed outcomes;
- forced move-policy rebind observed in active games;
- five setup epochs observed and timed;
- paired checkpoint/save/load continuation equivalence;
- h0 paired EMA export and digest re-verification;
- no search/training-opponent imports or participants;
- one injected supervisor stop;
- telemetry schema and append-resume continuity.

Do not run a strength comparison or start the 12-hour job.

## 10. Handoff and report

Deliver:

```text
reports/phase17/phase17_tandem_handoff_v1.json
reports/phase17/agent_04_report.md
reports/phase17/agent_04_schedule.json
reports/phase17/agent_04_resume_rehearsal.json
```

Bind all input handoffs, source/config/schedule/checkpoint/export schema digests,
measured iteration time, expected `N`, setup budget, persistence evidence, and guard
evidence. Set `ready_for_external_handshake` and `ready_for_preflight` separately.
