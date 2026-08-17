# Phase 9 — Agent 8
# Independent Final Acceptance and Phase 9 Freeze

## Mission

Perform the first and only sealed final evaluation of the frozen Phase 9 checkpoint.

No training, tuning, checkpoint replacement, threshold modification, or new experiment is permitted after the final-test bank is opened.

Recommend `PASS`, `FAIL`, or `BLOCKED`.

Formal Phase 9 acceptance remains with the reviewing chat.

## Prerequisites

Require Agents 1–7 `PASS` and formal acceptance.

Independently verify all identities from live bytes/source rather than trusting handoff prose.

### Mandatory corpus resolver check

Resolve the accepted Phase 8 corpus exclusively through:

```text
synthetic_corpus.default_corpus_root()
```

Require expected current path and all three accepted digests.

Do not hard-code the absolute path in evaluation library code.

A corpus mismatch is `BLOCKED`; do not regenerate/repair it.

## Independently verify

At minimum:

```text
rules
engine
observation
action frame
C1 parameter count
C1 config digest

Phase 8 anchor checkpoint SHA/model checksum

Phase 9 checkpoint SHA/model checksum
Phase 9 selected iteration

phase9_train_config_v1 document digest
trainer runtime identity digest if distinct

phase9_population_v1
phase9_rollout_schedule_v1
phase9_advantage_v1
phase9_checkpoint_v1
phase9_eval_bank_v1
phase9_acceptance_v1

Phase 7 setup library digest
Phase 9 validation bank digest
Phase 9 final-test bank digest
```

Verify final checkpoint parameters/logits finite before evaluation.

Verify training-discipline evidence:

```text
no final-test neural access before Agent 8
no post-test training
fresh Phase 8 start
pilot count exactly six
validation-only checkpoint selection
```

## Open sealed final-test bank

Only after all prerequisite checks pass.

Use greedy float32 `single_request` direct policy.

No search. No temperature sampling.

Use exact frozen color pairing and setup cases.

## A. Final vs Phase 8 anchor

Run all 512 paired cases:

```text
1,024 games
```

Require:

```text
EWR >= 0.58
paired 95% lower bound > 0.53
```

Record W/D/L by color, family, terminal reason, and game length.

## B. Strategic hard gate

Evaluate Phase 9 final and Phase 8 anchor on the exact same frozen Strategic final cases.

Require Phase 9:

```text
EWR >= 0.52
paired EWR improvement over Phase8 >= +0.05
95% CI lower bound for paired improvement > 0
```

Use the Agent 1-frozen paired-bootstrap unit and seed.

Report stretch target `EWR >= 0.55` separately; it is not a hard gate.

## C. Tactical hard gate

Same protocol:

```text
EWR >= 0.52
paired EWR improvement over Phase8 >= +0.05
95% CI lower bound for paired improvement > 0
```

Report `0.55` stretch separately.

## D. Random regression guard

Run all frozen final Random cases.

Require:

```text
overall EWR            >= 0.94
Red EWR                >= 0.90
Blue EWR               >= 0.90
paired 95% lower bound > 0.92

illegal actions         0
model failures          0
non-finite outputs      0
```

## E. Basic regression guard

Require:

```text
EWR                    >= 0.65
paired 95% lower bound > 0.60
```

## F. Belief retention

Use the accepted Phase 8 held-out synthetic belief benchmark and original remaining-count baseline.

Require:

```text
belief CE / remaining-count CE <= 0.98
belief top-1 > remaining-count top-1
```

Do not refit the baseline on test.

This is belief retention only. Phase 8 policy imitation CE is report-only.

## G. Policy-collapse/safety

Across final evaluation states report legal-policy entropy distribution and:

```text
fraction with max legal probability > 0.999
```

Require:

```text
< 0.25
```

Also require zero:

```text
illegal actions
observer-safety violations
non-finite policy/value/belief outputs
model failures
```

## Report-only evaluations

Run frozen report-only stress schedule.

Report:

```text
stress opponent EWRs
setup-family EWR
color EWR
terminal reasons
game length
policy entropy
value calibration
belief by piece type
belief by progress
archive cross-play / league matrix
Phase9 vs Phase8 by family
Strategic/Tactical paired deltas by family
```

Do not use these to rescue a failed hard gate.

## Final suite

After all Agent 8 artifacts and tests exist:

1. run targeted Agent 8 tests;
2. run the complete repository suite;
3. if artifact-gated tests require a second steady-state run, run it;
4. record exact passed/skipped/failed counts.

Any new failure blocks PASS.

## Artifacts

Create:

```text
reports/phase_9_data/agent_08_final_acceptance.json
reports/phase_9_data/agent_08_strength_results.csv
reports/phase_9_data/agent_08_league_matrix.csv
```

The acceptance JSON must include a machine-readable hard-gate table with observed value, threshold, and boolean result.

## Recommendation semantics

### PASS

Recommend PASS only if every hard gate passes and every identity/discipline gate is clean.

### FAIL

Recommend FAIL if the experiment was valid and a frozen performance/safety gate is missed.

Do **not** retrain or change thresholds after observing final test.

### BLOCKED

Recommend BLOCKED for invalid experiment identity, corruption, leakage, missing required evidence, or inability to execute the frozen evaluation faithfully.

## Completion gates

At minimum:

```text
agents1_7_pass
corpus_resolver_verified
corpus_digests_match
phase8_checkpoint_verified
phase9_checkpoint_verified
phase9_config_verified
final_bank_verified
pre_agent8_final_test_access_zero
phase9_vs_phase8_gate
strategic_gate
tactical_gate
random_gate
basic_gate
belief_retention_gate
collapse_gate
illegal_actions_zero
model_failures_zero
nonfinite_outputs_zero
observer_safety_zero
paired_bootstrap_exact
report_only_diagnostics_written
full_suite_green
```

## Forbidden

After final-test opening, do not:

- retrain;
- run a new pilot;
- change checkpoint;
- change threshold;
- change setup cases;
- change bootstrap method;
- change opponent implementation;
- add search;
- add learned setup selection;
- begin Phase 10.

## Final report statement

Conclude with exactly one recommendation:

```text
PHASE 9 RECOMMENDATION: PASS
```

or

```text
PHASE 9 RECOMMENDATION: FAIL
```

or

```text
PHASE 9 RECOMMENDATION: BLOCKED
```

Then stop.

Formal closure belongs to the reviewing chat.
