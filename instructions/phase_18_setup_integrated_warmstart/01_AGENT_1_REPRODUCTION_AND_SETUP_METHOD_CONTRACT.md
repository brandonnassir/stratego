# Phase 18 - Agent 1

## Reproduction boundary, setup-method parity map, and evaluation contract

## Mission

Make Phase 18 safe to begin. Establish the immutable research boundary, reconstruct
the exact accepted Phase 8 control, replace the Phase 17 paper-only setup
interpretation with a paper-plus-published-code method map, and freeze the evaluation
questions before any Phase 18 learner is implemented or trained.

Read `00_PHASE_18_ADAPTIVE_SEQUENCE_AND_COMMON_CONTRACT.md` completely. It governs
this task.

You do not build the setup model, modify the Phase 8 trainer, generate a live corpus,
run a Phase 8 control, or authorize production. Small read-only calculations and
non-learning parity fixtures are permitted. No meaningful optimizer step is permitted.

## 1. Required reading

Read the actual sources, not summaries alone:

```text
2511.07312v1.pdf

authors' published Stratego source at commit:
92db29e8ffc323b1b8a2804b5c3f84695d036b05
    pyengine/arrangement/buffer.py
    pyengine/arrangement/sampling.py
    pyengine/networks/arrangement_transformer.py
    pyengine/core/rl.py
    pyengine/core/train_container.py

instructions/phase_8_sequential_agent_plan/
reports/phase_8_implementation_report.md
reports/phase_8_data/

instructions/phase_17_tandem_current_policy_self_play/
reports/phase17/agent_05_report.md
reports/phase17/agent_07_report.md
reports/phase17/phase17_run_closeout_v1.json
reports/phase17/ataraxos_method_map_v1.md

stratego/training/phase17/setup_contract.py
stratego/training/phase17/setup_model.py
stratego/training/phase17/setup_sampling.py
stratego/training/phase17/setup_episode.py
stratego/training/phase17/setup_learning.py

stratego_project_docs/README.md
stratego_project_docs/05_project_plan.md
canonical status/evidence-index documents
```

If the published source is unavailable, stop and report the exact failure. Do not fall
back to the stale Phase 17 method map while calling it parity.

## 2. Establish the process and source boundary

Read-only verify and record:

- current Git commit and complete working-tree status;
- every modified/untracked Phase 17 evidence artifact used by Phase 18;
- active learner, evaluator, supervisor, dashboard, or monitor processes;
- current repository/run freeze state;
- available storage and the external Phase 8 corpus resolver;
- accepted Phase 8 checkpoint and corpus paths; and
- current test-suite status before Phase 18 edits.

Do not stop a process because it exists. Do not run `git clean`, reset user changes,
delete evidence, rewrite history, or make an unapproved commit.

Deliver:

```text
reports/phase18/phase18_process_boundary_v1.json
```

It must contain a timestamp, source identity, full status classification, active
process observations, external data paths, material artifact hashes, problems, and a
plain `source_boundary_ready` boolean.

If the source closure cannot be made immutable without an operator action, set it
false and provide the smallest exact operator action required.

## 3. Close the documentation gap

Update the canonical project status, evidence index, and plan only after verifying the
underlying artifacts. The update must state:

```text
Phase 17 complete
RUN-2026-B evaluated
no checkpoint promoted
move-only training result negative
joint result flat
setup model worse than fixed library in the pooled late window
Phase 18 planned but not yet authorized beyond Agent 1
```

Preserve historical Phase 17 instruction amendments. Do not rewrite them to reflect
the newly published implementation; document the difference in the Phase 18 method
map.

## 4. Reconstruct the exact Phase 8 control

Independently recompute or verify every frozen identity:

```text
historical source revision               53050b9
accepted checkpoint SHA-256              f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca
C1 config digest                          31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d
canonical fresh C1 state/checksum          from the accepted Agent 6/7 artifacts
corpus version                            synthetic_warmstart_corpus_v1
corpus content digest                     c95c3545b07f2341e7efbc83c79e6342510dd973038b0f72e7eae013cff87d0d
corpus metadata digest                    1db0f02fe45b16f539f070b1e12d4fdd6f390fd0487180fe660af0f4d49c81bb
commit-index digest                       32e8e18d1ca57ee555ed848851284f5938d4989ceb6c864f83ca4b9286c15db1
Phase 4 bank digest                       5fe5f98750ca2bd90ee75a74b3ba024bf753342872ae5472f13eb7afbb674266
setup library digest                      7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
train config digest                       3cab772bd8f74677efcdc1f90ec6f383490313f7652d82bd7fedf86153919ae7
trainer runtime identity                  64db92539a7d6c06ac4d01e4e904857da5b95c3d86d1287e108ede19e4f03879
```

Freeze the exact Phase 8 configuration from the accepted machine-readable artifacts;
do not retype it from memory. Record the accepted validation/test metrics and all 42
original completion gates.

Define the new-control comparison against the accepted checkpoint:

- exact paired cases and seeds;
- head-metric comparison and aggregation units;
- game-level paired bootstrap;
- confidence level and replicates;
- practical non-inferiority margins selected before the result; and
- how MPS nondeterminism is handled without weakening logical resume or data identity.

Deliver:

```text
reports/phase18/phase18_phase8_reproduction_contract_v1.json
```

Set `ready_for_phase8_control` true only if an independent agent can launch the control
without choosing a new field.

## 5. Build the setup-method parity map

Create:

```text
reports/phase18/ataraxos_setup_method_map_v2.md
reports/phase18/ataraxos_setup_method_map_v2.json
```

Use one row per method element:

```text
paper section/equation/table
paper text/constant
published-code file/line/behavior
Phase 17 behavior
required Phase 18 behavior
status: exact | scaled | corrected | intentional integration divergence | not used
reason
required unit/reduction/parity test
required telemetry
future owning agent
```

At minimum cover:

- setup sequence, masking, causal factorization, prefix alignment, and orientation;
- forced handedness and post-generation random reflection;
- W/D/L order and expected-value calculation;
- categorical averaging of repeated setup outcomes;
- suffix negative log likelihood and entropy-return recursion;
- Eq. 1 normalization and the operational `I - 10h` advantage residual;
- PPO ratio, clipping, reverse KL direction, and fixed coefficient;
- value/entropy/policy loss weights;
- setup pool generation, lifetime, reuse, and behavior-snapshot attribution;
- batch size, epochs, learning rate, optimizer semantics, gradient clip, and EMA;
- raw versus EMA actor roles;
- model-size scaling arithmetic;
- all Phase 8 integration divergences from paper self-play; and
- belief/search separation.

### 5.1 Required entropy derivation

The map must show rather than merely assert the correction.

Paper Eq. 1 trains:

```text
h_theta(sigma) approximately H(sigma_bar | sigma; theta_t) / 10
```

The published implementation stores normalized `h`, then executes:

```text
H_hat = reg_norm * h, reg_norm = 10
entropy residual = realized suffix NLL - H_hat
```

Therefore the Phase 18 sampled-path estimator is:

```text
I - 10h
```

Phase 17's `I - h` must be labeled a corrected implementation divergence. Do not call
the unit mismatch an intentional faithful transcription now that the implementation
is available.

### 5.2 Required pool/reward derivation

Show the data-flow distinction between:

```text
paper/published code:
sample pool -> many game resets -> aggregate completed outcomes by exact setup -> PPO

Phase 17:
sample fresh setup for a game -> one outcome -> immediate episode PPO
```

Define the Phase 18 setup identity and count/variance fields needed to reproduce the
former semantics.

## 6. Freeze setup architecture scaling

Recompute trainable parameter counts for:

```text
paper setup model
paper move model
project C1 move model
proposed Phase 18 setup model
```

Record:

```text
paper setup/move parameter ratio
project setup/move parameter ratio
proportional target
absolute and percentage difference
measured memory and forward-pass estimates if available without training
```

The governing default is 4/128/4/512 at 802,320 trainable parameters. A different
architecture requires a demonstrated gate failure and later operator decision.

## 7. Freeze the evaluation questions and packs

Create:

```text
reports/phase18/phase18_evaluation_contract_v1.json
```

Define all four factorial lanes in the common contract and the following setup strata:

```text
familiar
unusual_procedural
operator_sealed
setup_learning_development
```

The contract must specify:

- provenance and inclusion/exclusion rules;
- reflection-class de-duplication;
- train/development/sealed anti-overlap checks;
- opponents, colors, cases, rule versions, and exact seed derivation;
- overall, color, opponent, setup-family, and worst-stratum metrics;
- original Phase 8 sealed head metrics;
- unusual-setup belief metrics by ply/reveal bucket;
- paired bootstrap semantics;
- retry/failure semantics;
- selection versus final-test access rules; and
- immutable manifest/digest construction.

The operator's exact unusual setups are not required to be readable by Agent 1. The
contract must support an opaque sealed manifest whose identity and non-overlap can be
verified without exposing its contents to training or selection code.

## 8. Power and precision plan

Use Phase 17's per-case paired outcomes and observed variability to calculate the game
counts needed to detect the predeclared practical margins. Do not copy the 120-game
lane size by convention.

Report power/precision for:

- learned setup versus fixed library;
- trained setup versus fresh setup initialization;
- tandem C1 versus reproduced C1 on unusual opponent setups; and
- combined tandem system versus reproduced C1 plus fixed library.

When exact analytic power is inappropriate for W/D/L paired data, use a deterministic
simulation/bootstrap power study. Select margins before any Phase 18 candidate result.

## 9. Evaluator credibility requirements

Specify tests that a future evaluator agent must pass before it can open a sealed pack:

- identical deterministic results across supported worker counts;
- forced worker failure and retry without duplication or outcome substitution;
- missing result is `missing`/`failed`, never draw;
- exact case/seed/color/setup accounting;
- no setup or opponent orientation drift;
- lane isolation: changing own setup cannot change move weights or opponent case;
- model/setup digest refusal on mismatch;
- sealed pack cannot be opened by training or pilot paths; and
- full result recomputation from immutable row receipts.

The Phase 17 retry defect is a required regression case.

## 10. Agent 1 report and handoff

Create:

```text
reports/phase18/agent_01_report.md
reports/phase18/phase18_agent1_handoff_v1.json
```

The report leads with findings, not work performed. It must plainly state:

- whether Phase 8 is reproducible under a frozen contract;
- every material Phase 17-to-published-method correction;
- whether evaluation can distinguish Claim A, Claim B, and the combined system;
- every unresolved item and why it matters; and
- the recommended next experiment, without writing its implementation instruction.

The handoff must include independently verifiable digests and these booleans:

```text
source_boundary_ready
phase17_documentation_closed
phase8_control_contract_frozen
official_setup_method_map_complete
entropy_units_resolved
symmetry_contract_resolved
pool_and_outcome_aggregation_resolved
evaluation_contract_frozen
power_plan_frozen
evaluator_requirements_frozen
ready_for_phase8_control
ready_for_setup_parity_build
```

Absence is not pass. Set a readiness field true only when its evidence path and digest
are present.

## 11. Completion gates

Recommend `PROCEED` only if:

- the repository/source boundary is immutable and non-destructive;
- Phase 17 is documented as closed with no promotion;
- every Phase 8 identity recomputes;
- the exact control command/config can be constructed without a new choice;
- the paper/published-code/local method map has no material unresolved field;
- `I - 10h`, symmetry, pool reuse, and outcome averaging are explicit;
- evaluation packs and anti-leak rules are frozen;
- practical margins and required sample sizes are predeclared; and
- the next bounded question is identifiable.

If those pass, stop and deliver the decision packet. Do not begin the control run and
do not author an Agent 2 instruction inside this task.
