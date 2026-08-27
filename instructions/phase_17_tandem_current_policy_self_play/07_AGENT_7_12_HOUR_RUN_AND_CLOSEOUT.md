# Phase 17 — Agent 7
## Twelve-hour operation, external-result reconciliation, and checkpoint shortlist

## Mission

Operate exactly one authorized 12-active-hour Phase 17 tandem run, preserve the full
learning curve, enforce the frozen collapse policy without mid-run tuning, reconcile
all external evaluations, and recommend a robust paired checkpoint.

You may start only with a digest-valid Agent 6 GO record and explicit operator approval
of the launch time. You do not redesign code, change hyperparameters, add belief/search,
or substitute a new benchmark during the run. Read
`00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md` completely before validating GO.

## 1. Immediate prelaunch verification

Before creating h0:

- verify no learner/heavy evaluator already owns the training machine;
- recompute the launch/source/config/Phase 9/benchmark/environment identities;
- confirm the remote worker is awake, reachable, empty of conflicting candidate work,
  and using the accepted evaluator environment;
- confirm checkpoint, telemetry, export, result, and backup destinations have adequate
  free space;
- confirm power/awake safeguards on both computers;
- run the production command in validation/dry-run mode if provided;
- verify the supervisor is armed with the launch-manifest thresholds.

Any mismatch invalidates GO. Stop and return to Agent 6; do not patch the manifest.

## 2. Start and h0

Create the run only through the frozen Phase 17 start loader. Verify the exact Phase 9
raw move state, fresh optimizer/controller/schedule, fresh setup model, both initial
EMAs, and run ID.

Before the first optimizer update:

1. write and verify the initial joint checkpoint;
2. export the paired h0 EMA candidate;
3. enqueue it to external evaluation;
4. record every file/model/config digest in the run ledger.

If h0 identity does not match the launch manifest, stop immediately.

## 3. Twelve active hours

Run until active-training elapsed time reaches 12 hours. Planned downtime for a safe
exact resume does not count toward active time. Candidate nominal times derive from
active elapsed time, not wall-clock relabeling.

At each iteration:

- preserve the frozen transition/setup budgets, schedules, epoch counts, KL rules,
  population, setup pools, and queue policy;
- inspect supervisor status and high-level move/setup/system telemetry;
- never tune based on interim EWR;
- never replace a failed external case or benchmark stratum;
- allow observable non-silent engineering failures to be recorded and handled by the
  frozen policy.

At h0 and each 30-minute boundary through h12, create one immutable paired EMA export.
There should be 25 nominal candidates. If a boundary is delayed, record actual active
time and retain its original ordinal; do not fabricate or rename a candidate.

## 4. External-evaluation operation

Follow Agent 5's exact start/monitor/retry procedure. For every candidate ledger:

- bundle publication time and hashes;
- remote acceptance/refusal status;
- move-only and joint-lane completion times;
- returned receipt and verification;
- mean EWR and all frozen strata;
- backlog, retry, failure, or missing-result reason.

The trainer should normally continue while remote evaluation is healthy. The
training-side supervisor applies fixed-pack collapse rules from verified receipts.
Never enter an EWR manually or use a result lacking the candidate/pack receipt.

## 5. Stop and failure behavior

Apply common-contract section 13 exactly. Immediate integrity failures stop at once.
Persistent predicates use their stored consecutive counts; one noisy evaluation does
not stop the run.

On a stop:

1. cease new collection safely;
2. write the terminal checkpoint/telemetry/stop record if integrity permits;
3. preserve partial external work under its true status;
4. record the exact predicate, evidence rows, active time, and last valid identities;
5. do not restart under changed settings or attempt to finish 12 hours as the same run.

For an ordinary crash, resume only from an exactly verified Phase 17 joint checkpoint.
Confirm the next action/setup/update continuity evidence. If exact resume fails, stop;
do not reseat games or drop setup episodes invisibly.

## 6. Monitoring emphasis

The operator wants failures observed in production unless they could falsify results.
Keep monitoring compact and evidence-bearing:

- fixed transition counts and iteration time;
- current raw move digest on both seats;
- move LR, KL, entropy, clip fraction, and EMA identity;
- setup optimizer activity, five epoch count, KL, empirical/predicted entropy;
- reflection diversity, flag/bomb support, and setup queue age/backlog;
- game-length/terminal-reason distribution;
- remote cadence latency, receipt integrity, mean EWR, and worst stratum.

Do not add ad hoc probes that consume training RNG or alter machine contention.

## 7. Twelve-hour closure

At 12 active hours:

- stop only at the runner's defined safe boundary;
- write and verify the terminal hot checkpoint and h12 EMA export;
- wait for or explicitly account for every external candidate receipt;
- freeze the run ledger, telemetry, exports, results, stop state, and source/config
  identities;
- verify 25 candidate ordinals or explain each missing/failed ordinal;
- do not promote the h12 checkpoint automatically.

## 8. Learning-curve and robustness report

For every eligible candidate report:

- active hour and checkpoint/model identities;
- move-only and joint mean EWR;
- worst opponent/setup stratum EWR and its identity;
- opponent, setup, and color strata;
- three-point rolling medians;
- move/setup KL and entropy context;
- setup diversity/effective-support context;
- integrity or guard eligibility flags.

Answer directly:

1. Did the system reproduce useful early learning?
2. Did the hour 6–12 rolling curve continue upward, remain flat, or degrade?
3. Did mean improvement hide a worst-stratum regression?
4. Did the setup network remain legal, train continuously, and retain meaningful
   diversity?
5. Was external cadence complete and identity-clean enough to support selection?

No significance claim is required. Use honest engineering uncertainty and do not
compare EWR values from different pack versions.

## 9. Shortlist and recommendation

Exclude any candidate with an integrity failure, collapsed setup distribution, or
unresolved receipt mismatch. From hour 6–12 eligible candidates, construct a Pareto
shortlist emphasizing:

1. mean EWR;
2. worst-stratum EWR;
3. late rolling-curve direction;
4. move-only non-regression;
5. setup diversity and stability.

Recommend one paired checkpoint and up to two alternatives. Explain every tradeoff.
A single noisy peak or final timestamp cannot by itself win. The operator makes the
promotion decision; do not copy the recommendation into an accepted/promoted path.

## 10. Handoff and report

Deliver:

```text
reports/phase17/phase17_run_closeout_v1.json
reports/phase17/agent_07_report.md
reports/phase17/agent_07_learning_curve.csv
reports/phase17/agent_07_checkpoint_shortlist.json
```

Bind the full run, candidate, receipt, pack, source, config, guard, and shortlist
identities. State whether the run completed 12 active hours, why it stopped if not,
what was established, and what remains unknown.

Phase 17 ends here. Belief training, belief-guided search, and stochastic human-facing
selection require a separately authorized later-phase instruction package.
