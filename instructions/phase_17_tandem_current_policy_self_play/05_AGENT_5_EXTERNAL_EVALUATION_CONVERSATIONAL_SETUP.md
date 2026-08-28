# Phase 17 — Agent 5
## Conversational MacBook setup and external evaluation pipeline

## Mission

Work directly with the operator to determine whether their MacBook can evaluate paired
Phase 17 checkpoints every 30 minutes, then build and prove the safest practical
transport/evaluation/receipt workflow for the actual machines.

This is intentionally a conversational assignment. Do not assume a Mac model, SSH,
network topology, shared drive, installation permission, or preferred workflow. Do
not launch the 12-hour training run. Read
`00_PHASE_17_SEQUENCE_AND_COMMON_CONTRACT.md`,
`09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md`, and the Agent 1 evaluation
schema before speaking as though a design has been selected.

## 1. First interaction with the operator

Begin with a short explanation of what must complete within each 30-minute interval:
transfer, digest verification, two benchmark lanes, and result receipt. Then collect
facts in small conversational batches rather than presenting a long questionnaire.

Resolve at least:

- exact MacBook model/year, chip, CPU cores, RAM, macOS, and free storage;
- whether it can remain powered, awake, cooled, and mostly idle for at least 13 hours;
- Python/PyTorch environment, architecture compatibility, and permission to create an
  isolated environment if needed;
- how the MacBook and training computer can communicate during the run;
- host/account access, firewall/VPN/NAT constraints, and allowed authentication;
- whether direct push, remote pull, shared storage, or a manual-assisted path is
  acceptable;
- safe CPU worker count and other operator constraints.

Ask only what is needed for the next decision. Explain any command the operator must
run and interpret its result with them. Never request passwords, private keys, or
secrets in chat. Prefer key-based or locally approved authentication where available.

Record confirmed facts and unresolved items in
`reports/phase17/agent_05_remote_discovery.json`; do not invent missing values.

## 2. Feasibility decision

Propose the simplest reliable design consistent with the facts. Consider, in order:

1. direct immutable bundle push plus result return;
2. MacBook polling/pulling from an authenticated source;
3. shared local storage with atomic publication;
4. a carefully specified manual handoff only for the initial gate—not unattended
   production.

Explain security and reliability tradeoffs to the operator and obtain their choice
before changing either machine. If unattended 30-minute operation is infeasible, say
so and propose the smallest change that makes it feasible. Do not quietly downgrade
the cadence or omit the joint lane.

## 3. Portable evaluator

Agent 4 has frozen the export schema at `c2c0365`. Implement a remote evaluator for
the D10 `RUN-2026-B` paired candidates under additive Phase 17 namespaces. It must:

- accept only an atomically published immutable bundle;
- verify expected files, file SHA-256, move/setup state digests, architecture, rules,
  config, source, candidate time, and composite-pack identity before loading;
- run both `move_only` and `joint_move_setup` lanes exactly as Agent 1 froze them;
- use EMA weights only;
- use fixed cases/seeds and produce overall plus opponent/setup/color strata;
- be idempotent for a candidate identity;
- never select a mutable `latest` checkpoint;
- refuse a partial, duplicate-conflicting, stale-attributed, or incompatible bundle.

The move-only lane still records the paired setup digest with `setup_used: false`.
The joint lane samples the candidate setup EMA from its fixed setup seeds and uses the
candidate move EMA for play.

Package dependencies in the operator-approved manner. Bind the exact source commit and
environment identity; local uncommitted code must not be reconstructed manually on the
MacBook.

## 4. Transport and receipt protocol

Implement the selected design with:

- staging under a `.partial` or equivalent non-final identity;
- transfer completion and local fsync where appropriate;
- remote hash verification before atomic publication;
- a per-candidate lock/idempotency record;
- explicit queue/backlog/retry status;
- results written atomically;
- a returned receipt containing candidate, both model states, pack, evaluator, host,
  runtime, lane result, result-file hash, and receipt digest.

No result is eligible until its receipt has been re-verified on the training computer.
Never relabel a delayed result with a later nominal cadence.

## 5. Benchmark identity

Consume Agent 1's immutable composite manifest. Re-verify the accepted Phase 16 lane
and the new fixed-seed joint lane on both machines. A portability change that affects
cases, rules, seeds, inference math, or scoring requires a new pack/evaluator version;
do not preserve a digest across semantic changes.

Run a small cross-machine identity fixture before the full pack. For deterministic
fixed inputs, compare loaded model identities, legal moves, logits/probabilities within
the frozen tolerance, generated setup tokens under fixed seeds, and final scoring.
Platform floating-point differences must be measured and documented, not assumed
bit-identical.

## 6. Early-candidate gate and cadence rehearsal

With the operator, transfer one D10 paired h0 candidate and complete the full workflow:

1. source bundle manifest and hash;
2. transport staging and atomic publication;
3. remote identity verification;
4. both evaluation lanes;
5. atomic results and receipt return;
6. source-side receipt verification and ledger ingestion.

Do not turn this into a repeated performance-certification exercise. One complete
identity-clean round trip is the required silent-false-result check. Report transfer,
verification, each lane, and receipt times and state plainly whether the observed path
fits the 30-minute cadence. If it does not, keep backlog explicit rather than changing
the pack or attributing a late result to a later candidate.

## 7. Failure behavior

The remote worker may retry transport or evaluation safely, but it may not change the
benchmark, worker count, inference mode, model weights, or candidate identity after
the gate. Failures generate explicit receipts/status rows. Backlog remains bound to
the original candidates. Disk exhaustion, host sleep, network loss, identity mismatch,
and result-return failure must be visible to Agent 7.

The remote worker does not directly stop the trainer. It returns identity-bound facts;
D10 treats finite EWR decline as learning-curve telemetry rather than a stop command.

## 8. Handoff and report

Deliver:

```text
reports/phase17/phase17_external_eval_handoff_v1.json
reports/phase17/agent_05_report.md
reports/phase17/agent_05_remote_discovery.json
reports/phase17/agent_05_roundtrip_receipt.json
reports/phase17/agent_05_cadence_rehearsal.json
```

The handoff binds the operator-approved topology without exposing secrets, remote host
and environment identity, bundle/receipt schemas, composite pack/evaluator digests,
latencies, queue behavior, and the exact start/monitor/stop procedure Agent 7 will use.
Set `ready_for_unattended_30m_evaluation` true only after the operator agrees and the
full early-candidate round trip passes.
