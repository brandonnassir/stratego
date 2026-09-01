# Phase 17 — Operator Decision D11
## Export during training; evaluate afterward on the Mac Mini

_Written 2026-08-28 and revised before launch after the operator prioritized starting
training as quickly as possible._

## 1. Decision and precedence

Phase 17 will not run evaluation while the 12-hour trainer is active. The trainer will
still publish immutable paired EMA candidates at h0 and every 30 active minutes through
h12, preserving the full learning curve. After training stops and all candidates are
frozen, Agent 5 will evaluate them locally on the same Mac Mini.

No MacBook, SSH, network share, remote worker, cross-computer transfer, remote receipt,
or prelaunch h0 evaluation is required for `RUN-2026-B`.

This decision supersedes every Phase 17 requirement for live, external, remote, or
concurrent evaluation; MacBook discovery; cross-machine comparison; transport; an h0
evaluation gate; or evaluator readiness before launch. D10's training recipe, fixed
benchmark semantics, immutable paired EMA candidates, 30-active-minute export cadence,
identity checks, and hour 6–12 selection rule remain unchanged.

## 2. Training-time boundary

During training:

- the trainer writes one atomically published immutable paired EMA candidate at each
  frozen cadence boundary;
- every candidate binds run, ordinal, active time, move/setup state, source, config,
  and export identities;
- no evaluator process runs and no candidate is scored;
- no evaluation queue, worker, receipt, EWR, or benchmark result is a launch or runtime
  dependency; and
- training never pauses or competes for resources with evaluation.

An export failure or identity mismatch remains an integrity failure because it would
destroy the intended learning curve. The absence of an evaluation result during
training is expected and is not a warning.

## 3. Post-training evaluation

After Agent 7 stops at the safe h12 boundary, verifies the terminal checkpoint, and
freezes the 25 candidate ordinals, Agent 5 resumes its local evaluator work. It first
validates one frozen candidate end to end, then evaluates all candidates sequentially
using both required lanes:

1. `move_only` with fixed boards/setups and evaluation opponents; and
2. `joint_move_setup` with fixed setup RNG cases and paired EMA move/setup weights.

The evaluator verifies candidate, model-state, pack, config, source, evaluator, host,
and environment identities before loading. It writes idempotent local results and
receipts outside the training checkpoint/telemetry directories. Delayed, retried, or
failed results remain bound to their original candidate ordinals.

Because the trainer is no longer running, there is no concurrent-resource experiment,
safe-boundary pause mode, or 30-minute evaluation-runtime gate. Evaluation may be
repaired or resumed after training without changing the preserved run.

## 4. Revised agent sequence

The governing sequence is:

```text
Agent 4C attribution/resume correction
    -> Agent 6 short launch freeze
    -> Agent 7 twelve-hour training and candidate freeze
    -> Agent 5 local evaluation, learning curve, and checkpoint shortlist
    -> operator promotion decision
```

Agent 5's existing uncommitted source-side evaluator and composite benchmark work may
be preserved for later reuse. Its remote discovery and transport work are not launch
inputs and must not be committed into the training source merely to finish the retired
remote workflow.
