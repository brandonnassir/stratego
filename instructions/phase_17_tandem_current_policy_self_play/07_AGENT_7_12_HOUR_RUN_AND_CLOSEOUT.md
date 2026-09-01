# Phase 17 — Agent 7
## Twelve-hour training operation and candidate freeze

## Mission

Operate exactly one authorized `N=640` Phase 17 tandem run (approximately 12 active
hours), preserve its training telemetry and 24 or 25 immutable paired EMA candidates,
and freeze the completed run for Agent 5. Do not evaluate candidates while training is
active.

Start only with a digest-valid Agent 6 GO record and explicit operator approval. Do not
redesign code, change hyperparameters, add belief/search, or substitute participants.
Read `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md`,
`09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md`, and
`11_OPERATOR_DECISION_D11_LOCAL_EVALUATION.md` before validating GO. Treat
`reports/phase17/phase17_launch_manifest_v2.json` as the controlling launch record,
including conditions A6-C1 through A6-C4.

## 1. Immediate prelaunch verification

Before creating h0:

- verify no learner or evaluator process already owns the Mac Mini;
- recompute launch/source/config/Phase 9/environment identities;
- confirm checkpoint, telemetry, export, and backup destinations have adequate space;
- confirm power and awake safeguards;
- run the frozen production command in dry-run mode;
- use the manifest's `phase9_move_start` and `fresh_setup_start` blocks to verify the
  two identities omitted by `--describe`;
- verify the supervisor's integrity stops and statistical warnings; and
- verify evaluation and transport processes are absent from the launch command.

Any mismatch invalidates GO. Return to Agent 6; do not patch the manifest in place.

## 2. Start and h0

Create the run only through the frozen Phase 17 start loader. Verify exact Phase 9 raw
move weights; fresh move optimizer, controller, schedule and EMA; newly random setup
model; fresh setup optimizer and EMA; recipe `phase17_simple_paper_tandem_v1`; and run
ID `RUN-2026-B`. Load no Agent 3/4 setup training state.

Before the first optimizer update:

1. write and verify the initial joint checkpoint if the frozen launch path requires it;
2. export and verify the immutable paired h0 EMA candidate; and
3. record all file/model/source/config digests in the run and candidate ledgers.

Do not score h0. If its identity does not match the launch manifest, stop immediately.
The required h0 EMA state digests are `f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd`
for the move model and `9dc73986f4e31f0654c8432ecf49eea001d75c34aa58975ea63dfb7c1e5207aa`
for the setup model.

## 3. Twelve active hours

Run the frozen `N=640` iterations. This was sized for approximately 12 active hours;
depending on production throughput it may finish about 61 seconds short of the exact
h12 boundary and produce 24 candidates instead of 25. A 24-candidate result is valid,
not an integrity failure. Do not extend past `N=640` to manufacture h12 because the
move LR and entropy schedules are frozen to this horizon. Planned downtime for an exact
resume does not count toward active time. Candidate nominal times derive from active
elapsed time, not wall-clock relabeling.

At each iteration:

- preserve the fixed move-transition budget, five setup epochs, fixed setup reverse-KL
  coefficient, shared-iteration alpha schedule, population, and fresh setup pools;
- consume every setup episode completed in that iteration exactly once;
- inspect supervisor status and high-level move/setup/system telemetry;
- never tune or restart because interim training telemetry looks poor; and
- allow visible non-silent engineering failures to be handled by the frozen policy.

At h0 and every 30-active-minute boundary reached through h12, create one immutable
paired EMA export. The frozen run should yield 24 or 25 candidate ordinals under
A6-C1. For each export immediately verify its file hash, manifest digest, move/setup
EMA digests, source/config identity, ordinal, and active time. Do not invoke Agent 5 or
any evaluator.

If a boundary is delayed, retain its original ordinal and record actual active time.
Never fabricate, overwrite, rename, or silently omit a candidate.

## 4. Stop and failure behavior

Apply D10 section 7 exactly. Stop for identity/routing/legality/numerical/persistence,
prohibited-participant, fixed-transition, export-integrity, or unrecoverable-resource
failures. Treat finite KL, entropy, diversity, concentration, and game-length behavior
as warnings and experiment results.

On a stop:

1. cease new collection safely;
2. write the terminal checkpoint, telemetry, and stop record if integrity permits;
3. preserve all candidates already published under their true identities;
4. record the predicate, evidence rows, active time, and last valid identities; and
5. do not restart under changed settings as the same run.

For an ordinary crash, resume only from an exactly verified joint checkpoint. Confirm
next-action/setup/update continuity. If exact resume fails, stop; do not reseat games,
drop setup episodes, or repair telemetry invisibly.

## 5. Monitoring emphasis

Keep monitoring compact:

- fixed transition counts and iteration time;
- current raw move digest on both seats;
- move LR, KL, entropy, clip fraction, and EMA identity;
- setup optimizer activity, five epoch count, fixed reverse KL, alpha,
  empirical/predicted entropy, and completed episodes;
- setup diversity and flag/bomb support as descriptive telemetry;
- game lengths and terminal reasons;
- checkpoint/resume identity; and
- candidate export count, ordinal, active time, and digests.

There is no live EWR, evaluator latency, receipt, or evaluation backlog during training.
Do not add probes that consume training RNG or machine resources.

## 6. Twelve-hour closure

At the frozen `N=640` terminal boundary:

- stop only at the runner's safe boundary;
- write and verify the terminal checkpoint and the h12 EMA export if the active-time
  boundary was reached;
- verify 24 or 25 candidate ordinals and record whether the last ordinal is h11.5 or
  h12; account explicitly for any count below 24;
- freeze the run ledger, telemetry, checkpoints, exports, stop state, and source/config
  identities;
- confirm no trainer process remains active; and
- hand the frozen candidate ledger to Agent 5 without promoting h12 automatically.

Report training facts only: whether 12 active hours completed, integrity/warning state,
iteration and game throughput, optimizer activity, late move/setup telemetry direction,
and candidate completeness. Make no strength claim and no checkpoint recommendation.

## 7. Handoff and report

Deliver:

```text
reports/phase17/phase17_run_closeout_v1.json
reports/phase17/agent_07_report.md
reports/phase17/agent_07_training_telemetry_summary.csv
reports/phase17/agent_07_candidate_ledger.json
```

Bind the full run, source/config/start, checkpoint, telemetry, candidate and guard
identities. Set `ready_for_post_training_evaluation: true` only when the trainer is
stopped and every candidate ordinal is frozen or explicitly accounted for.

Agent 5 then performs local evaluation, learning-curve analysis, and checkpoint
shortlisting. Belief training, belief-guided search, and stochastic human-facing move
selection remain separate later work.
