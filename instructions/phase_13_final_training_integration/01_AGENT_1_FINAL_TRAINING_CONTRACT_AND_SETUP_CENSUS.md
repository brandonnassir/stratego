# Phase 13 — Agent 1
## Final Training Contract and Setup Census

## Mission

Resolve and freeze every training-system decision that must be known **before** the 168-hour Phase 14 run, then perform the one short setup-distribution census required to make sure the production setup source is safe to freeze.

This is an engineering-contract task. It is **not** a reinforcement-learning experiment, hyperparameter sweep, new belief-model study, search experiment, large setup research project, strength tournament, or reinterpretation of any closed earlier phase.

Do not start Phase 14.

## 1. Preserve the accepted project history

Do not overwrite, delete, reinterpret, or mutate accepted artifacts from earlier phases. Preserve at minimum:

```text
accepted Phase 9 checkpoint
Phase 10 / P10-D artifacts and closure
all Phase 11 evidence and formal FAIL
all Phase 11B engineering artifacts
Agent 1C checkpoint
all Phase 12 search artifacts
phase12_search_candidate_v1
```

Phase 11 remains:

```text
phase11_final_classification = FAIL
phase11_reinterpreted = false
```

Phase 11B remains an engineering branch only. Phase 12 search remains outside Phase 14 training.

## 2. Starting model for Phase 14

The Phase 14 direct-policy training run starts from the **accepted Phase 9 C1 checkpoint**.

Use:

```text
accepted Phase 9 C1
    policy
    value
    accepted auxiliary objectives
```

Do **not** use Agent 1C as the policy/value starting checkpoint. Agent 1C is preserved for the later search/belief pipeline only.

## 3. Retrieve the accepted Phase 9 training configuration

Read the actual accepted Phase 9 configuration and implementation. Do not reconstruct numerical values from memory.

Record the exact accepted values for at least:

```text
optimizer family
accepted Phase 9 learning rate
any Phase 9 LR schedule
policy objective
value objective
belief auxiliary objective, if present
belief auxiliary weight, if present
ratio clipping
behavior-policy/KL regularization
advantage filtering
gradient clipping
EMA behavior
batch/update settings
trajectory format
game rules
any other loss coefficients required for exact continuation
```

Bind the source artifact/config identity for each retrieved value. If a value is absent from the accepted artifacts, state that explicitly rather than guessing.

## 4. Conservative continuation learning rate

Phase 14 is continuation/fine-tuning, not another high-LR relearning run.

Let:

```text
LR9 = exact accepted Phase 9 learning rate
```

Default proposal:

```text
main continuation LR = 0.25 × LR9
late continuation LR = 0.125 × LR9
```

You may choose a different **single conservative continuation pair** only if accepted Phase 9 or other already-accepted project evidence clearly supports it. If adjusted, document the accepted evidence, rationale, final multiplier, and final exact LR.

Forbidden:

```text
LR sweep
multiple candidate LRs
rehearsal comparisons across LRs
changing LR from live Phase 14 metrics
raising LR during Phase 14
```

Freeze one main LR and one late LR.

## 5. Main / late schedule

Define the schedule conceptually as:

```text
first ~75–80%
    main continuation segment

final ~20–25%
    late historical-heavy segment
    lower continuation LR
```

Then freeze an exact wall-clock transition time before Phase 14. The previous 132h / 36h split may be used if no better already-accepted engineering rationale exists, but it is not sacred.

Record:

```text
main segment exact hours
late segment exact hours
transition elapsed time
transition UTC derivation rule
```

The transition is tied to **original Phase 14 wall-clock elapsed time**, not optimizer steps. Downtime counts against the schedule and does not move the transition.

## 6. Final opponent mixture

Handcrafted opponents must remain a regularizer, not dominate the expensive run.

Freeze one simple mixture with:

```text
neural current + historical combined = 85–90%
rule/unusual combined = 10–15%
```

Preferred starting proposal:

### Main segment

```text
58% current neural policy
30% historical neural opponents
12% handcrafted/diversity opponents
```

Handcrafted 12%:

```text
3% Strategic
3% Tactical
2% Scout-rush
2% Miner-rush
2% Information-miser
```

### Late segment

```text
40% current neural policy
48% historical neural opponents
12% handcrafted/diversity opponents
```

Use the same five handcrafted opponent families unless an already-accepted implementation constraint requires a nearby equivalent. Do not perform an opponent-mixture sweep. Freeze the exact main and late percentages.

## 7. Historical archive vs active historical pool

These are separate concepts.

### Historical archive

Every 2 hours of original Phase 14 elapsed wall-clock time, preserve a durable model snapshot. All durable snapshots remain in the archive. No tournament is required for archive admission.

### Active historical pool

The active pool is the bounded subset actually sampled during self-play.

Target:

```text
16 neural historical opponents total
```

Permanent anchors:

```text
Phase 8
accepted Phase 9
```

Up to 14 Phase 14 snapshots.

Preferred age composition once enough snapshots exist:

```text
older Phase 14 snapshots        4
middle-aged Phase 14 snapshots 4
recent Phase 14 snapshots       6
```

Preferred sampling weights within the historical share:

```text
permanent anchors 20%
older             25%
middle            25%
recent            30%
```

Freeze the exact deterministic membership/update algorithm. Requirements:

- based only on archive ordering/age;
- no tournament for admission;
- deterministic when categories are not yet full;
- deterministic redistribution when an age category is unavailable;
- exact active-pool membership and category state stored in hot checkpoints;
- crash/resume reconstructs the exact same active pool;
- no uniform sampling from all accumulated archive snapshots.

Record the algorithm precisely enough that Agent 2 can implement it without interpretation.

## 8. Preserve the belief auxiliary objective

Inspect the accepted Phase 9 learner. If it includes a supervised belief auxiliary objective:

```text
retain the same objective
retain the same accepted weight
retain the same target construction
```

Do not remove or downweight it merely because Agent 1C will later receive dedicated belief specialization. If it is not present, record that fact. If an implementation defect prevents its use, stop and report the defect rather than silently changing the training objective.

## 9. Search is forbidden in Phase 14 training

Freeze:

```text
Phase 12 TINY search   NOT USED
Phase 12 SMALL search  NOT USED
Phase 12 MEDIUM search NOT USED
```

Search may not be used for self-play action selection, training targets, opponent policies, policy improvement, or trajectory generation. Phase 14 trains the direct C1 policy/value system only.

## 10. Checkpoint hierarchy

Freeze three checkpoint roles.

### Hot resume checkpoints

Cadence:

```text
every 15 minutes
```

Keep at least the most recent four valid hot checkpoints on fast internal storage.

Each must include enough state to resume exactly:

```text
model weights
optimizer state
EMA state
optimizer/global step
RNG / deterministic stream state
population schedule state
active historical pool
historical archive cursor/state
trajectory/shard cursor
storage state required for safe continuation
original Phase 14 start time
absolute Phase 14 deadline
main/late schedule state
candidate-evaluation scheduling state
```

### Durable historical / archive checkpoints

Cadence:

```text
every 2 hours
```

Store durably, preferably on the external training volume.

### Final-policy candidate checkpoints

Cadence:

```text
every 6 hours
```

Include:

```text
hour 0
hour 6
hour 12
...
hour 162
hour 168
```

These may be a marked subset of the 2-hour archive snapshots. The hour-168 checkpoint is a candidate, not automatically the deployed final policy.

## 11. Fixed candidate evaluation pack

Create one fixed direct-policy engineering evaluation pack before Phase 14.

Do not use search. Do not use the spent Phase 11 sealed test bank.

Recommended size:

```text
128 games per candidate
```

Balanced across:

```text
32 vs accepted Phase 9
32 vs Strategic
32 vs Tactical
32 vs Scout-rush
```

Freeze boards, colors, setups, opponent seeds, player seeds, rules, and evaluation implementation. Every candidate checkpoint uses the exact same pack.

This evaluation is monitoring/selection infrastructure only. It may never change ongoing Phase 14 training. It cannot stop training early, change LR, change opponent mixture, change setup source, change historical-pool logic, change checkpoint cadence, or extend the deadline.

If candidate evaluation fails during Phase 14, preserve the candidate, continue training, and evaluate it later using the same frozen pack.

## 12. Predeclared post-run checkpoint selection rule

Freeze the final-policy selection rule before Phase 14.

### Primary

Highest equal-weight mean EWR across:

```text
accepted Phase 9
Strategic
Tactical
Scout-rush
```

### Tie-break 1

Highest:

```text
minimum opponent-stratum EWR
```

### Tie-break 2

If still exactly tied:

```text
later checkpoint
```

No confidence intervals or statistical tests are required. No post-run reweighting.

At hour 168:

```text
training stops
    ↓
complete any missing fixed-pack candidate evaluations
    ↓
apply the frozen rule
    ↓
select final direct-policy checkpoint
```

Selection time is outside training time and may not include additional optimizer updates.

## 13. Absolute 168-hour wall-clock contract

Freeze:

```text
run_start_utc = timestamp immediately before Phase 14 training loop begins
run_deadline_utc = run_start_utc + 168 hours
```

Persist both in every hot checkpoint.

On restart:

```text
remaining_time = run_deadline_utc - current_time
```

Never create a fresh 168-hour deadline. Downtime counts against the run. The main/late transition is also based on original wall-clock time.

## 14. Exact deadline behavior

Freeze the optimizer-boundary semantics.

At or after the original deadline:

1. stop launching new collection units;
2. finish or safely discard the active bulk unit according to the existing accepted bulk-sync boundary;
3. do not begin a new optimizer step after the deadline;
4. write the final run-state checkpoint;
5. preserve/mark the hour-168 candidate;
6. write final counters and manifest;
7. mark Phase 14 training closed.

If a recovery attempt starts after the deadline, the runner must refuse further optimizer steps and finalize.

## 15. Short deterministic exposed-Flag setup census

Perform one narrow census expected to take minutes, not hours. Do not play a large match set. Do not reopen Phase 10.

### Sources

Sample separately from:

```text
neutral_v1
P10-D learned branch
final proposed 35/65 production mixture
```

Recommended minimum:

```text
10,000 deterministic setups per source
```

If paired-army geometry is needed for immediate-Scout exposure, generate a deterministic paired-start sample large enough to measure it cleanly.

### Required measurements

Report at minimum:

```text
Flag row distribution
Flag location distribution
forwardmost deployment-row Flag rate
immediately Scout-accessible Flags
one-move/trivially exposed Flag patterns
games where a Flag can be captured before that defender receives a decision
source branch responsible
pre-perturbation vs post-perturbation
pre-reflection vs post-reflection
whether perturbation creates/amplifies exposure
whether reflection creates/amplifies exposure
```

Also identify whether exposure is associated primarily with neutral generation, learned selection, reflection, perturbation, or interaction between the two armies' starting layouts.

### Alarm criteria must be written before sampling

Before opening census results, write `setup_census_alarm_policy` distinguishing:

#### Defect

Any rule/contract violation, coordinate transformation error, invalid placement, unintended Flag movement, or reflection/perturbation implementation error. A defect requires repair before Phase 14.

#### Pathology

Define a narrow quantitative threshold for a clearly pathological rate of trivial pre-play / immediate Scout Flag captures in the final production mixture. Choose and record the threshold and rationale **before sampling**. Do not change it after seeing results.

#### Valid but strategically poor

A legal distribution that places Flags aggressively or poorly but does not meet the predeclared defect/pathology criteria. Report it, but do not automatically reopen Phase 10.

## 16. Setup repair rule

If no defect/pathology is found:

```text
phase14_setup_source_v1
    35% neutral_v1
    65% accepted P10-D learned selector
    accepted reflection/perturbation behavior
```

If a narrow implementation/setup-distribution defect is found:

- repair only the production wrapper/source needed for Phase 14;
- create a new Phase 13/14 setup-source identity;
- preserve all closed Phase 10 evidence unchanged;
- document exactly what changed and why;
- rerun only the census checks necessary to verify the narrow fix.

Do not retrain the Phase 10 utility model. Do not conduct setup-strength experiments.

Freeze `phase14_setup_source_v1` after this check.

## 17. Storage / retention contract

Use existing storage evidence as the starting point. Earlier soak rate was approximately:

```text
3.572 GiB/hour
```

A naive 168-hour raw archive could approach ~600 GiB.

Inspect the actual intended external volume. Project:

```text
raw trajectory growth
+ hot checkpoints
+ 2-hour archive snapshots
+ logs
+ candidate evaluations
+ other Phase 14 artifacts
+ 20% safety reserve
```

If full raw retention does not fit, freeze a rolling retention policy before launch. A valid policy may retain all checkpoints, all metrics/summaries, all historical snapshots, representative trajectory shards, and delete only already-consumed disposable Phase 14 raw shards.

Never delete earlier accepted project evidence to create space.

## 18. Monitoring without tuning

Freeze monitoring for:

```text
elapsed wall-clock
remaining wall-clock
optimizer step
games generated
positions generated
collection throughput
learner throughput
policy loss
value loss
belief auxiliary loss if present
gradient norm
learning rate
advantage-filter acceptance fraction
draw rate
game length
current/historical opponent mix
active historical pool
archive size
checkpoint age
disk usage
worker health
non-finite counters
candidate evaluation status
```

The control interface must not make it convenient to change frozen training values during Phase 14. At minimum prevent normal live edits to LR, loss weights, opponent mixture, setup source, historical-pool algorithm, candidate-selection rule, and deadline. Emergency stop remains available.

## 19. Deliverables

Create at minimum:

```text
phase13_final_training_contract_v1
phase13_setup_census_alarm_policy_v1
phase13_setup_census_v1
phase14_setup_source_v1
phase14_checkpoint_selection_pack_v1
phase14_checkpoint_selection_rule_v1
phase13_agent_01_report.md
phase13_agent_01_summary.json
```

The final training contract must contain all exact resolved values needed by Agent 2.

## 20. Stop condition

Stop after:

- accepted Phase 9 values are retrieved;
- continuation LR is frozen;
- main/late schedule is frozen;
- opponent mixtures are frozen;
- historical archive/active-pool rules are frozen;
- belief auxiliary treatment is frozen;
- checkpoint hierarchy is frozen;
- candidate pack and selection rule are frozen;
- setup alarm policy is written;
- setup census is completed;
- any narrow setup defect is resolved if required;
- final Phase 14 setup source is frozen;
- storage/deadline/recovery semantics are frozen.

Do not run RL training. Do not begin Agent 2 automatically.
