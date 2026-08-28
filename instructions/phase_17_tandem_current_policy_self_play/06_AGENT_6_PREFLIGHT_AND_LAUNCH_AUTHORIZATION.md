# Phase 17 — Agent 6
## Short launch-integrity check

## Mission

Confirm that the exact D10 build will produce attributable rather than silently false
results, then issue GO or NO-GO. This is not a model-certification phase. Read the
common contract and `09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md`; D10 governs.

The complete local check is capped at 30 minutes. Reuse Agent 2/4 evidence for retained
infrastructure and Agent 4B's targeted tests. Do not rerun their broad suites.

## 1. Inputs

Require:

- Agent 4B handoff with `ready_for_short_launch_check: true`;
- exact source commit and `phase17_simple_paper_tandem_v1` config digest;
- exact Phase 9 checkpoint file/model-state digests;
- a production command that initializes a fresh setup model;
- fixed external benchmark identity; and
- one external h0 identity round trip or an operator-confirmed plan to complete it
  immediately before production h0.

## 2. One end-to-end check

Run only the shortest real production-path smoke needed to verify:

1. Phase 9 move weights load with fresh move training state;
2. setup model, optimizer, and EMA start from scratch under a recorded seed;
3. both seats sample legal moves from the same current raw move policy;
4. both sides receive fresh legal setups from the current raw setup policy;
5. the collector emits the exact fixed-transition budget;
6. completed games update every associated setup episode exactly once for five epochs;
7. setup PPO clip is `0.2`, Adam LR is `5e-5`, reverse-KL coefficient is fixed `0.1`,
   EMA is `0.999`, and alpha is `0.1 * global_iteration^-0.3`;
8. checkpoint save/load preserves identities and neither loses nor duplicates a
   completed setup outcome; and
9. search, belief, historical, handcrafted, and argmax training paths are absent.

Discard the smoke weights. Production must start again from exact Phase 9 plus a newly
random setup model.

## 3. Do not do

Do not run a standalone setup soak, diversity gate, EWR strength test, parameter sweep,
controller calibration, queue study, schedule study, extended resume campaign, or
failure-injection matrix. Do not block on finite KL, entropy, diversity, or early EWR
behavior; those are production telemetry.

## 4. Launch record

Emit a compact immutable record:

```text
reports/phase17/phase17_launch_decision_v2.json
reports/phase17/phase17_launch_manifest_v2.json
reports/phase17/agent_06_report.md
```

Bind the run ID `RUN-2026-B`, source/config/Phase 9/fresh-setup identities, production
command, fixed benchmark and external worker identity, nine smoke results, artifact
destinations, and h0-through-h12 export cadence.

Decision:

```text
GO     all nine silent-false-result checks passed
NO-GO  an identity, routing, legality, numerical, persistence, or prohibited-path
       check failed
```

There is no setup-diversity or learning-quality gate. Explicit operator approval is
still required to start the 12-hour run.
