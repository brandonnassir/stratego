# Phase 17 — Agent 6
## Short launch freeze

## Mission

Confirm that the exact D10 build will produce attributable rather than silently false
results, then issue GO or NO-GO. This is not a model-certification phase. Read the
common contract, `09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md`, and
`11_OPERATOR_DECISION_D11_LOCAL_EVALUATION.md`; D10 governs training and D11 governs
evaluation.

The complete local check is capped at 15 minutes. Reuse Agent 4B's accepted D10 smoke
and Agent 4C's focused evidence. Do not rerun a production-shaped smoke or broad suite.

## 1. Inputs

Require:

- Agent 4C handoff `phase17_simple_tandem_handoff_v2` with non-empty source identity,
  lossless resume telemetry, valid checkpoint ancestry, and
  `ready_for_launch_freeze: true`;
- exact source commit and `phase17_simple_paper_tandem_v1` config digest;
- exact Phase 9 checkpoint file/model-state digests;
- a production command that initializes a fresh setup model; and
- the immutable h0-through-h12 paired export cadence and output destinations.

## 2. Evidence and command check

Do not train. Verify only:

1. Agent 4B's nine D10 smoke results and source-file hashes still match their accepted
   artifacts;
2. Agent 4C's focused tests and correction identities match the current source; bind
   code correction commit `67b186a` and record the current administrative/report HEAD
   separately if it differs;
3. the dry-run command reports `RUN-2026-B`, fresh setup initialization, a non-empty
   source digest, the frozen schedule digest, and 25 candidate ordinals; recompute the
   config digest from `would_run` with the production digest function and require
   `4e7abad83dab4a9bec1cf82f4b238059eb9de0c344bf0025f91aa0a703b2d1b7`;
4. the production output directory is new, destinations have adequate free space, and
   no learner process is active;
5. h0 is written before either optimizer update and later exports are driven by active
   training time; and
6. no evaluator, transport worker, search, belief, historical opponent, or handcrafted
   opponent appears in the production command or training participant configuration;
   and
7. every modified or untracked path outside the 22-file production closure is listed
   explicitly and cannot be imported by the training entry point. Do not require the
   unrelated operator-plan or deferred Agent 5 evaluator work to be deleted.

The telemetry schema still labels its legacy evaluation-status field
`not_connected_in_this_session`. Under D11 this means evaluation is deliberately
deferred until after training; record that interpretation in the launch manifest and
do not treat it as a missing live evaluator.

## 3. Do not do

Do not run a tandem smoke, evaluator check, standalone setup soak, diversity gate, EWR
test, parameter sweep, controller calibration, queue study, schedule study, extended
resume campaign, or failure-injection matrix. Evaluation is explicitly post-training.

## 4. Launch record

Emit a compact immutable record:

```text
reports/phase17/phase17_launch_decision_v2.json
reports/phase17/phase17_launch_manifest_v2.json
reports/phase17/agent_06_report.md
```

Bind the run ID `RUN-2026-B`, source/config/Phase 9/fresh-setup identities, production
command, accepted Agent 4B smoke plus Agent 4C correction evidence, artifact
destinations, and h0-through-h12 export cadence. Record that Agent 5 evaluation begins
only after Agent 7 freezes the run.

Decision:

```text
GO     the accepted smoke, Agent 4C corrections, identities, and launch command match
NO-GO  an identity, routing, legality, numerical, persistence, or prohibited-path
       check failed
```

There is no setup-diversity or learning-quality gate. Explicit operator approval is
still required to start the 12-hour run.
