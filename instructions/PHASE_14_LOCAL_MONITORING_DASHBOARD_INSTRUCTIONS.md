# Phase 14 Local Read-Only Monitoring Dashboard Instructions

## Mission

Build a lightweight local monitoring dashboard for the frozen **168-hour Phase 14 training run**.

The dashboard is an **observational convenience only**. It is not part of the training algorithm and must not become a second control plane.

The central architectural rule is:

```text
Phase 14 trainer / supervisor
        ↓
approved telemetry, logs, store manifests, checkpoint metadata
        ↓
read-only dashboard
```

Never:

```text
dashboard
    ↓
trainer configuration or training state
```

If the dashboard crashes, freezes, is closed, or is never opened, **Phase 14 training must continue completely unaffected**.

## 1. Scope

Build a small local web dashboard that can be opened occasionally in a browser during Phase 14.

Recommended access pattern:

```text
http://127.0.0.1:<port>
```

Bind only to localhost unless explicitly instructed otherwise.

The dashboard should be simple, fast, low-overhead, and easy to inspect at a glance.

Do not turn this into a polished general-purpose application or a new engineering phase.

## 2. Read-only requirement

The dashboard must be strictly read-only.

It may obtain information only from already-approved Phase 14 sources such as:

- training telemetry files;
- supervisor state and event logs;
- rollout-store manifests;
- checkpoint metadata;
- archive/candidate metadata;
- existing run status files;
- filesystem/disk-space information;
- existing structured logs.

It must **not**:

- modify trainer state;
- modify checkpoints;
- modify population state;
- modify setup selection;
- modify the absolute deadline;
- change any frozen configuration;
- write commands into the learner;
- become required for recovery;
- create a new control path into Phase 14.

Emergency and recovery actions must continue to use the accepted **Phase 14 supervisor and runbook**.

## 3. Training isolation

The dashboard must not consume meaningful training resources.

### Forbidden work

Do not:

- import or invoke the model for dashboard purposes;
- run neural-network inference;
- use MPS/GPU for dashboard work;
- run search;
- run candidate strength evaluations;
- run tournaments;
- run the full test suite while training;
- reconstruct trajectories unnecessarily;
- perform expensive database/log scans on every refresh.

### Resource target

Aim approximately for:

```text
CPU:
near-zero when idle

Memory:
tens to low hundreds of MB

MPS/GPU:
zero

Disk:
light read-only access

Browser refresh:
roughly every 5–15 seconds while viewed
```

Prefer event summaries, cached telemetry, or incremental log reads over repeatedly parsing large files.

If practical, reduce refresh work when no browser client is connected.

## 4. Main dashboard

The first screen should answer these questions immediately:

1. Is Phase 14 alive?
2. Is it operationally healthy?
3. How far through the 168-hour run are we?
4. Is training advancing?
5. Are checkpoints being written?
6. Are workers healthy?
7. Is storage healthy?
8. Have there been crashes, restarts, or loader-pool rebuilds?

Show at minimum:

| Field | Purpose |
|---|---|
| Overall run status | TRAINING / RECOVERING / FINALIZING / COMPLETE / ERROR |
| Run start UTC | Immutable Phase 14 start |
| Deadline UTC | Immutable 168-hour deadline |
| Elapsed time | Wall-clock time since start |
| Remaining time | Deadline minus current time |
| Run progress | Percentage of 168 hours completed |
| Schedule segment | MAIN or LATE |
| Learning rate | Current frozen scheduled LR |
| Optimizer step | Current training step |
| Population iteration | Current 2,048-game iteration |
| Games generated | Source from authoritative rollout-store manifests |
| Games/hour | Recent generation throughput |
| Learner throughput | Recent optimizer/update throughput |
| Loader workers | Live / configured workers |
| Loader-pool rebuilds | Count and latest rebuild |
| Supervisor restarts | Process restart count |
| Latest hot checkpoint | Timestamp / age |
| Latest archive | Timestamp |
| Latest candidate checkpoint | Candidate hour/timestamp |
| External drive free space | Remaining capacity |
| Nonfinite count | NaN/Inf safety signal |

Use the authoritative durable source for each value rather than counters known to under-report after restart.

## 5. 168-hour progress display

Include a prominent representation of total wall-clock progress.

For example:

```text
PHASE 14

████████████████░░░░░░░░░░░░░░░░  43.7%

73h 24m elapsed
94h 36m remaining

Current segment: MAIN
Late segment begins: hour 132
Absolute stop: hour 168
```

The dashboard must derive remaining time from the **original immutable deadline**, not from accumulated active-training time.

Downtime is lost time.

Never present a restarted 168-hour clock.

## 6. Main/late schedule visibility

Clearly show the frozen schedule transition:

```text
MAIN
0–132 hours
LR: 7.5e-5

LATE
132–168 hours
LR: 3.75e-5
```

Also show the current population segment if available.

The dashboard must display the frozen values only.

It must not provide controls to change them.

## 7. Training telemetry

Provide lightweight recent-history views for useful training signals.

Suggested metrics:

- policy loss;
- value loss;
- belief auxiliary loss;
- gradient norm;
- learning rate;
- advantage-filter acceptance fraction, if emitted;
- game-generation throughput;
- learner/update throughput;
- draw rate;
- average game length;
- nonfinite/NaN counts.

A 6–24 hour recent window is sufficient for ordinary viewing.

Do not interpret normal short-term loss movement as a model-quality failure.

The dashboard should present telemetry, not perform autonomous tuning decisions.

## 8. Operational health panel

Separate **system health** from **training metrics**.

Example:

```text
Supervisor          HEALTHY
Learner             HEALTHY
MPS                 HEALTHY
Loaders             6 / 6
Checkpoint age      7 minutes
External storage    HEALTHY
Pool rebuilds       0
Supervisor restarts 0
Nonfinite values    0
```

Use clear green/yellow/red status indicators only for operational conditions grounded in existing Phase 14 contracts.

Do not invent alarms such as:

```text
"policy loss increased, therefore training is bad"
```

unless such a threshold is already part of the frozen production contract.

## 9. Loader and recovery visibility

Because Phase 13 discovered a real loader-worker failure mode, loader status should be explicit.

Show:

```text
configured loader workers
live loader workers
loader-pool rebuild count
timestamp of latest rebuild
```

If a worker dies and the approved recovery logic rebuilds the pool, the event should appear in the dashboard without requiring manual log inspection.

The dashboard itself must not rebuild the pool.

That remains trainer/supervisor behavior.

## 10. Supervisor visibility

The external supervisor can see failures the learner cannot record after a hard kill.

Show supervisor information such as:

- supervisor alive/dead;
- learner process alive/dead;
- current process PID if useful;
- restart count;
- last exit signal/code;
- last restart timestamp;
- checkpoint resumed from;
- latest supervisor event.

Do not make the dashboard the supervisor.

It only reads the supervisor's durable state/events.

## 11. Checkpoint panel

Show the three relevant checkpoint concepts separately.

### Hot resume checkpoint

Expected cadence:

```text
15 minutes
```

Show:

- latest timestamp;
- age;
- path or short identifier if useful;
- whether it is readable/recognized by existing metadata.

### Archive checkpoint

Expected cadence:

```text
2 hours
```

Show latest archive time and identifier.

### Final-policy candidate

Expected cadence:

```text
6 hours
```

Show the most recent candidate checkpoint and candidate hour.

Do not evaluate candidates from the dashboard.

## 12. Storage panel

Show the actual external Phase 14 storage volume.

At minimum:

- mounted/unmounted;
- path;
- writable status if already exposed safely;
- total capacity;
- free space;
- used space;
- recent Phase 14 storage growth if inexpensive to compute.

Avoid recursively scanning the full training archive every few seconds.

Use filesystem metadata and existing Phase 14 accounting wherever possible.

## 13. Event log

Include a compact recent operational event stream.

Examples:

```text
09:00 Phase 14 launched
09:15 hot checkpoint written
10:02 population iteration 1 committed
11:00 archive checkpoint written
11:17 loader worker exited
11:17 loader pool rebuilt successfully
11:18 six loaders healthy
12:00 candidate checkpoint marked
```

Important events include:

- launch;
- schedule transition;
- checkpoint writes;
- archive writes;
- candidate marks;
- loader failures;
- loader-pool rebuilds;
- learner crashes;
- supervisor restarts;
- resume events;
- emergency stop;
- deadline reached;
- finalize-only closeout;
- completion.

Use bounded recent history in the browser rather than loading an unlimited log.

## 14. Forbidden UI controls

Do not include buttons or editable fields for:

- learning rate;
- opponent mixture;
- setup selector;
- batch size;
- loss weights;
- historical-pool strategy;
- search mode;
- training duration;
- absolute deadline;
- optimizer settings;
- checkpoint deletion;
- new training run;
- restart with a fresh 168-hour clock.

Prefer a status message such as:

```text
Training stopped — consult PHASE_14_RUNBOOK.md
```

rather than creating new recovery behavior inside the dashboard.

## 15. Independence requirement

Prove that the dashboard is not required for training.

At minimum verify:

1. trainer launches and operates with dashboard absent;
2. dashboard launches after trainer is already running;
3. killing the dashboard does not affect trainer or supervisor;
4. closing the browser does not affect training;
5. dashboard restart reconnects to the current telemetry without modifying training;
6. dashboard does not hold locks needed by trainer, supervisor, checkpoints, or rollout storage.

## 16. Code isolation

Keep the dashboard outside the frozen training import path wherever practical.

Treat it as an **operator/monitoring utility**, not training code.

Prefer an isolated location such as a monitoring or tools area rather than modifying the core Phase 14 learner merely to support presentation.

If additional telemetry fields are absolutely required, do not modify frozen training behavior casually. Prefer consuming existing outputs.

Any change that would alter the sealed Phase 14 code identity must be reported before launch-package sealing.

## 17. Localhost and security

Default to:

```text
127.0.0.1
```

Do not expose the dashboard on the public network or Internet by default.

No authentication system is necessary for a localhost-only passive monitor unless the existing project environment requires one.

Do not add cloud dependencies.

The dashboard should remain usable even if Internet access is unavailable.

## 18. Browser behavior

The browser UI should be lightweight.

Recommended:

```text
refresh while open:
5–15 seconds

charts:
recent bounded history only

animations:
minimal

network:
localhost only
```

The dashboard server may remain running for the seven-day session while the browser page is opened only intermittently.

## 19. Monitoring agent compatibility

The dashboard should make it easy for a human or monitoring agent to read the current state without performing expensive diagnostics.

A monitoring agent may inspect the dashboard or its underlying read-only status endpoint intermittently, but it remains subject to the Phase 14 monitoring rules:

- observe;
- report;
- follow the frozen runbook for approved recovery;
- never tune;
- never extend the deadline;
- never alter the experiment.

## 20. Verification before Phase 14

Build and verify the dashboard **before the Phase 14 launch**.

Required checks:

```text
dashboard starts locally
dashboard reads current approved telemetry
dashboard displays authoritative game counts
loader health is live
supervisor failures/restarts are visible
checkpoint ages are correct
deadline/elapsed/remaining calculations are correct
disk status is correct
no MPS/model inference occurs
no frozen training state is modified
trainer does not depend on dashboard
killing dashboard leaves trainer unaffected
```

Do not start Phase 14 as part of this task.

## 21. Deliverables

Produce:

```text
local Phase 14 dashboard implementation
short README / launch instructions
dashboard verification report
```

The report should state:

- what sources each displayed metric reads from;
- measured dashboard CPU/memory overhead;
- confirmation of zero MPS/model use;
- confirmation that the dashboard is read-only;
- confirmation that training continues if the dashboard is killed;
- localhost address/port;
- any known display limitations.

## 22. Stop condition

Stop when:

```text
dashboard implementation complete
read-only isolation verified
resource overhead acceptable
training independence verified
instructions documented
Phase 14 NOT STARTED
```

Return the implementation and verification results for review.

Do not launch the 168-hour training run.

# Design principle

The dashboard should answer:

> **Is training alive, healthy, progressing, checkpointing, and staying within its immutable seven-day contract?**

It should **not** answer that question by changing anything.

The safest dashboard is one that can disappear entirely and Phase 14 never notices.
