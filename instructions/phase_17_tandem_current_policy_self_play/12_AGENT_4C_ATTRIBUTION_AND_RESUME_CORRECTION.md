# Phase 17 — Agent 4C
## Narrow attribution and resume correction

## Mission

Starting from implementation commit `3be8bba`, correct the small provenance and resume
defects found in review. Do not change the move model, setup model, optimizer equations,
schedules, sampling, transition budget, setup epochs, population, or D10 recipe. This
is not another tandem build or preflight.

Read `09_OPERATOR_DECISION_D10_SIMPLIFIED_PAPER_TANDEM.md` and
`11_OPERATOR_DECISION_D11_LOCAL_EVALUATION.md`. The local evaluator in Agent 5 must be
able to reject a candidate that is not bound to the exact corrected source.

The working tree also contains operator plan edits and Agent 5's uncommitted evaluator,
benchmark, report, and retired transport work. Preserve them exactly. Do not delete,
rewrite, stage, or commit those unrelated paths as part of 4C. Their presence does not
authorize scope expansion.

## 1. Required corrections

1. Make the production source-closure digest mandatory and non-empty. Bind it into the
   production command, dry-run description, run digest, checkpoints, paired exports,
   and resume validation. Do not hard-code the pre-correction closure; Agent 6 will
   supply the final frozen digest through the launch manifest.
2. Preserve the telemetry row belonging to the checkpointed iteration across resume.
   The current checkpoint-before-append ordering loses one valid row per resume. Keep
   or replace that ordering as simply as possible, but prove that a checkpointed row is
   retained exactly once and genuinely uncheckpointed later rows are still truncated.
3. After resume, make the next checkpoint/export name the checkpoint actually loaded
   as its parent. Do not copy the loaded checkpoint's parent and skip a generation.
4. Include cadence-generated P4–P7 warnings in the iteration telemetry row when they
   trip. Preserve the established first-hour move-entropy baseline across resume.
5. Treat any rejected completed setup episode as an integrity stop. A duplicate,
   already-consumed, or incomplete outcome may be recorded for diagnosis, but training
   may not silently continue after dropping it.
6. Correct handoff provenance. Distinguish the starting/evidence-parent commit
   `eab8a33`, implementation commit `3be8bba`, this correction commit, and the final
   recomputed source-closure digest.

## 2. Targeted verification only

Add or update focused tests proving:

- an empty production source digest is refused;
- source identity changes the run identity and survives checkpoint/export/resume;
- checkpoint -> telemetry append -> resume retains every checkpointed iteration row
  exactly once;
- the next checkpoint after resume links directly to the loaded checkpoint;
- a cadence warning appears in the corresponding telemetry row;
- the first-hour entropy baseline survives resume; and
- a rejected completed setup episode arms the integrity stop.

Do not rerun the production-shaped D10 smoke, the broad Phase 17 suite, the repository
suite, a setup soak, or any strength evaluation. Run only the directly affected test
modules or individual tests.

## 3. Handoff

Deliver:

```text
reports/phase17/agent_04c_report.md
reports/phase17/phase17_simple_tandem_handoff_v2.json
```

The amended handoff binds the final correction commit/source closure, production
command shape, focused test results, and the five corrected invariants. Set
`ready_for_launch_freeze: true` only when production can create immutable cadence
exports with a non-empty exact source identity and resume cannot silently alter the
learning curve.
