# Phase 11B Closure Agent

## Mission

Finish only the already-running Phase 11B Agent 4 hybrid experiment, preserve its result, formally close the engineering sprint, and prepare the project for Phase 12.

Do not launch Phase 11B Agent 5.

Regardless of Agent 4's result, the selected Phase 12 belief model is Agent 1C.

## 1. Finish existing Agent 4 work only

Allow the already-running hybrid raw+C1 experiment to complete exactly as currently planned.

Preserve its checkpoint, architecture/config, parameter count, learning curve, training wall-clock, time-to-best, development metrics, standardized summary, and report.

Do not lengthen the experiment. Do not add additional evaluations. Do not rerun earlier Phase 11B candidates.

## 2. Do not launch Agent 5

Record:

```text
phase11b_agent5_status = cancelled_by_instructor
```

Do not implement, train, benchmark, or partially initialize it.

## 3. Selected candidate

Select Agent 1C for Phase 12 engineering.

Known result:

```text
R_CE = 0.9460
top-1 = 0.2640
```

Record the exact checkpoint path, checkpoint SHA/digest, model-state digest if available, architecture/config, frozen/unfrozen parameter description, training corpus identity, and development corpus identity.

Describe Agent 1C as:

> A copy of the accepted Phase 9 C1 in which the larger belief MLP and final C1 block were fine-tuned for supervised belief prediction.

## 4. Preserve formal history

Do not modify accepted Phase 9, Phase 10, or Phase 11 artifacts.

Explicitly record:

```text
phase11_final_classification = FAIL
phase11_reinterpreted = false
phase11_test_bank_spent = true
```

## 5. Phase 11B selection artifact

Create:

```text
phase11b_engineering_selection_v1
```

with:

```text
selected_candidate = Agent1C
selected_candidate_R_CE = 0.9460
selected_candidate_top1 = 0.2640
selection_type = engineering
scientific_claim = none
phase11_final_classification = FAIL
phase11_reinterpreted = false
phase11_test_bank_spent = true
phase11b_agent4_result = recorded
phase11b_agent5_status = cancelled_by_instructor
all_candidate_artifacts_preserved = true
```

Include the concise interpretation:

- dedicated supervised belief training produced the dominant gain;
- enlarging the head helped only modestly;
- allowing the final C1 block to adapt produced the best result;
- raw-observation and C1-feature CNNs were worse despite being larger;
- C1's representation appears highly useful and sample-efficient at the current data scale.

Agent 4's result must remain in the leaderboard for future reference even though it cannot change selection.

## 6. Phase 12 handoff

Prepare a concise handoff containing:

```text
accepted Phase 9 checkpoint identity
Agent 1C checkpoint identity
original Phase 11 belief identity
remaining-count baseline identity
accepted sampler interface
Phase 11B selection artifact identity
```

State:

```text
Phase 9 C1 -> policy/value
Agent 1C   -> belief only
```

Do not assume Agent 1C policy/value outputs are production-compatible after belief fine-tuning.

## 7. Stop condition

Stop after:

- Agent 4's existing result is preserved;
- Phase 11B leaderboard is updated;
- Agent 5 is marked cancelled;
- `phase11b_engineering_selection_v1` is created;
- Phase 12 handoff is written.

Do not begin search implementation.
