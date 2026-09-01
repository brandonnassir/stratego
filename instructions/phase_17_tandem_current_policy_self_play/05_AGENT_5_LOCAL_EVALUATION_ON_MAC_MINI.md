# Phase 17 — Agent 5
## Post-training local evaluation and checkpoint shortlist

## Mission

After the Phase 17 training run is complete, evaluate every frozen paired EMA candidate
locally on the Mac Mini, reconstruct the learning curve, and recommend a robust paired
checkpoint. You are not a launch dependency and do not run while training is active.

Read `00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md`,
`09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md`, and
`11_OPERATOR_DECISION_D11_LOCAL_EVALUATION.md`. D11 governs every conflict with the
former MacBook, transport, live-cadence, or h0-gate workflow.

## 1. Entry conditions

Begin only after Agent 7 delivers a frozen run closeout containing:

- the exact run/source/config/Phase 9 identities;
- the terminal checkpoint and telemetry identities;
- all expected h0-through-h12 paired candidate ordinals, or an explicit accounting of
  every missing/failed export; and
- confirmation that no trainer process remains active.

Preserve the existing uncommitted source-side evaluator and composite benchmark work
if it is valid. Ignore or remove from the active path all SSH, remote endpoint, MacBook,
transport, and cross-machine requirements. Do not discard existing work merely because
its original transport target was retired.

## 2. Evaluator contract

The evaluator must:

- accept only atomically published immutable paired candidates;
- verify expected files, file SHA-256, move/setup state digests, architecture, rules,
  config, source, candidate time, evaluator source, and benchmark-pack identity;
- load EMA weights only;
- run both `move_only` and `joint_move_setup` lanes exactly as Agent 1 froze them;
- use fixed cases/seeds and report overall plus opponent/setup/color strata;
- remain idempotent for a candidate identity;
- refuse partial, conflicting duplicate, stale-attributed, or incompatible artifacts;
- never read mutable raw weights or a mutable `latest` candidate; and
- write results and receipts outside training checkpoint/telemetry directories.

The move-only lane records the paired setup digest with `setup_used: false`. The joint
lane samples the candidate setup EMA from its fixed setup seeds and uses the candidate
move EMA for play.

## 3. One-candidate validation, then the batch

Before scoring all candidates, run one frozen candidate end to end and verify:

1. candidate/source/config/pack/evaluator identities;
2. both lane completions;
3. deterministic case/seed accounting;
4. atomic result and receipt writing; and
5. receipt re-verification and ledger ingestion.

This is a post-training evaluator check, not a model-quality gate. Fix visible evaluator
failures if needed without modifying candidate weights, cases, seeds, scoring, or the
preserved training run.

Once the check passes, evaluate all frozen candidates sequentially. A retry repeats the
exact same evaluation and never changes workers, inference mode, benchmark semantics,
weights, or candidate identity.

## 4. Learning curve and selection

For every eligible candidate report:

- active hour and checkpoint/model identities;
- move-only and joint mean EWR;
- opponent, setup, and color strata;
- worst-stratum EWR and identity;
- three-point rolling medians;
- move/setup KL and entropy context from the frozen training telemetry;
- setup diversity/effective-support context; and
- integrity or receipt eligibility flags.

Answer directly whether performance improved, flattened, or degraded from hour 6 to
hour 12 and whether mean improvement hid a worst-stratum regression.

Construct a Pareto shortlist from eligible hour 6–12 candidates emphasizing mean EWR,
worst-stratum EWR, late rolling direction, move-only non-regression, and setup stability.
Recommend one paired checkpoint and up to two alternatives. A single noisy peak or the
final timestamp cannot win automatically. The operator makes the promotion decision.

## 5. Handoff and report

Deliver:

```text
reports/phase17/phase17_local_eval_handoff_v1.json
reports/phase17/agent_05_report.md
reports/phase17/agent_05_local_environment.json
reports/phase17/agent_05_candidate_receipts.jsonl
reports/phase17/agent_05_learning_curve.csv
reports/phase17/agent_05_checkpoint_shortlist.json
```

Bind the preserved run, all candidate/result/receipt identities, benchmark/evaluator
source, failures/retries, learning curve, and shortlist. State what was established and
what remains unknown. Do not promote or overwrite an accepted checkpoint.
