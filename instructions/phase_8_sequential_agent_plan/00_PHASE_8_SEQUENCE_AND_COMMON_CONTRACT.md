# Phase 8 Sequential Agent Plan — Synthetic Warm-Start Training

## 1. Mission

Phase 8 builds and validates the project's first **meaningfully trained neural model**.

The project plan defines Phase 8 as:

```text
Generate games from the rule-based population and train policy/value/belief outputs.

Gate:
the model decisively beats the random baseline and learns nontrivial
value/belief predictions.
```

The broader project plan places **population self-play in Phase 9**, not Phase 8.

Accordingly, Phase 8 is a supervised / outcome-supervised synthetic warm start:

```text
frozen Phase 4 rule-based policy population
                    +
frozen Phase 7 setup sampler
                    ↓
deterministic compact synthetic game corpus
                    ↓
trajectory reconstruction
                    ↓
policy / WDL value / hidden-piece belief targets
                    ↓
C1 warm-start training on MPS
                    ↓
held-out evaluation
```

Phase 8 must leave the repository with:

1. a versioned synthetic-corpus contract;
2. a reproducible train/validation/test corpus built from rule agents;
3. a versioned training-example contract;
4. independently audited policy/value/belief targets;
5. a real MPS trainer for C1;
6. optimizer + scheduler + data-cursor checkpoint/resume;
7. one predeclared, bounded pilot-selection process;
8. one fresh canonical C1 warm-start run;
9. one accepted warm-start checkpoint;
10. evidence that the checkpoint beats random decisively and predicts value/belief nontrivially;
11. a clean handoff to Phase 9 population self-play.

Phase 8 is **not** the official 168-hour training run.

## 2. Source basis and relationship to the project

The accepted project plan states that Stage A should generate games from multiple rule-based agents with different styles and use them to warm-start move policy behavior, W/D/L value prediction, and hidden-piece belief prediction. Its purpose is to avoid spending a large fraction of the later 168-hour run learning elementary Stratego behavior from nearly random self-play.

The project plan separately places current-policy and historical-checkpoint population self-play in Phase 9, dynamically damped RL in the self-play stage, learned setup selection later, full belief validation later, and decision-time search later. Phase 8 must preserve those boundaries.

## 3. Sequential agent structure

Run these agents strictly in order.

| Agent | Responsibility |
|---|---|
| 1 | Freeze warm-start data, target, teacher-population, split, baseline, metric, and pilot-selection contracts **before corpus generation or training** |
| 2 | Generate, persist, resume, finalize, and independently verify the deterministic synthetic rule-agent corpus |
| 3 | Build the compact-to-example reconstruction pipeline; independently audit policy/value/belief targets and anti-leak boundaries |
| 4 | Build the C1 MPS trainer, optimizer/checkpoint/resume system, metrics, and throughput/stability benchmark |
| 5 | Run the bounded predeclared pilot matrix using train + validation only; freeze one exact `warmstart_train_config_v1` |
| 6 | From a fresh canonical C1 initialization, run the production Phase 8 warm start and freeze the selected checkpoint using validation only |
| 7 | Independently evaluate the frozen checkpoint on the sealed test corpus and frozen Phase 4 random evaluation; recommend PASS/FAIL/BLOCKED and freeze the Phase 9 handoff |

No later agent may begin until every prerequisite agent reports `PASS`.

No individual agent may formally close Phase 8. Agent 7 recommends a status; the reviewing chat makes formal acceptance.

## 4. Shared report

Create and maintain:

```text
reports/phase_8_implementation_report.md
```

Owned sections:

```text
# Phase 8 Implementation Report

## 1. Agent 1 — Warm-Start Contract and Pre-Training Acceptance Standard
## 2. Agent 2 — Synthetic Rule-Agent Corpus
## 3. Agent 3 — Training Examples, Targets, and Anti-Leak Audit
## 4. Agent 4 — MPS Trainer, Checkpoint/Resume, and Throughput Validation
## 5. Agent 5 — Bounded Pilot Selection
## 6. Agent 6 — Canonical C1 Warm-Start Run
## 7. Agent 7 — Independent Held-Out Evaluation and Phase 8 Freeze
```

Agent 1 creates the report header if absent. Later agents append only their section. Do not rewrite accepted earlier sections except for an explicitly authorized factual correction. Preserve historical descriptions when a later correction supersedes them.

Every headline result in Markdown must also exist in a machine-readable artifact.

## 5. Canonical Phase 8 artifacts

Required:

```text
reports/phase_8_data/agent_01_warmstart_contract.json
reports/phase_8_data/agent_01_teacher_population.json
reports/phase_8_data/agent_01_acceptance_thresholds.json

reports/phase_8_data/agent_02_corpus_manifest.json
reports/phase_8_data/agent_02_corpus_audit.json
reports/phase_8_data/agent_02_matchup_counts.csv

reports/phase_8_data/agent_03_example_contract.json
reports/phase_8_data/agent_03_target_audit.json
reports/phase_8_data/agent_03_validation_baselines.json

reports/phase_8_data/agent_04_trainer_contract.json
reports/phase_8_data/agent_04_training_benchmark.csv
reports/phase_8_data/agent_04_resume_validation.json

reports/phase_8_data/agent_05_pilot_runs.csv
reports/phase_8_data/agent_05_pilot_selection.json
reports/phase_8_data/agent_05_frozen_train_config.json

reports/phase_8_data/agent_06_warmstart_run.json
reports/phase_8_data/agent_06_training_curve.csv
reports/phase_8_data/agent_06_checkpoint_manifest.json

reports/phase_8_data/agent_07_heldout_metrics.json
reports/phase_8_data/agent_07_random_evaluation.json
reports/phase_8_data/agent_07_final_acceptance.json
reports/phase_8_data/agent_07_phase9_handoff.json
```

Large corpus shards and checkpoints are production data, not report attachments.

Preferred paths:

```text
data/warmstart/synthetic_warmstart_corpus_v1/
checkpoints/phase8/
```

The paths may be redirected to the external volume by configuration. Record the actual absolute/relative storage location and free space. Do not pretend an unavailable external drive was tested.

## 6. Frozen upstream state

Phase 8 begins only after formal Phase 7 acceptance.

The following are immutable upstream inputs:

```text
rules                         stratego_project_v1
reference engine              phase2_1_reference_1.2.0
observation                   observation_v2_1_127ch
engine action encoding        source_destination_10000_v1
engine action frame           absolute engine squares

model contract                model_contract_v2
model action frame            perspective_normalized_squares

primary model                 C1
C1 width                      128
C1 blocks                     4
C1 heads                      4
C1 FF width                   512
C1 trainable parameters       863,959
C1 config digest              31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d

fallback model                C0
C0 parameters                 123,223

backend                       KEEP_PYTHON
trajectory                    trajectory_v1
snapshot interval             32 where production recording is used

Phase 4 evaluation bank       evaluation_setup_bank_v1
Phase 4 bank digest           5fe5f987...674266
Phase 4 pairing/statistics    frozen

setup generator contract      setup_generator_contract_v1
setup families                setup_family_v1
setup trait schema            setup_trait_vector_v1
setup diversity standard      setup_diversity_standard_v1
setup seed derivation         setup_library_seed_v1
setup library                 setup_library_v1
setup perturbation            setup_perturbation_v1 / seed_encoding_v1
setup sampler                 setup_sampler_v1
setup source                  setup_source_v1
setup provenance              setup_provenance_v1

setup master seed             20260813
setup library entries         8,000
setup train/val/test          6,400 / 800 / 800
setup library digest          7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777

Phase 8 setup profile         neutral_v1
family sampling               uniform
base sampling                 uniform within family/split
reflection probability        0.5
perturbation probability      0.5
perturbation intensity        uniform swaps 1..6
perturbation Hamming window   [2,12]
perturbation retry budget     64
```

Do not modify `stratego/engine/`. Do not modify the frozen Phase 7 library, sampler semantics, splits, or Phase 4 evaluation bank. Any semantic change requires review and a new version, not a silent Phase 8 edit.

## 7. Phase 8 model-output semantics

C1 remains:

```text
input observations     [B,127,10,10]
legal mask             [B,10000] supplied separately

policy logits          [B,10000]
policy frame           perspective-normalized source/destination

value logits           [B,3]
value order            WIN, DRAW, LOSS
value perspective      acting/model player

belief logits          [B,100,12]
belief supervision     unresolved hidden opponent pieces only
belief square frame    perspective-normalized model/token frame
```

Privileged hidden-piece identities are targets only.

No true opponent setup, family, base id, perturbation seed, teacher-private state, or belief label may enter the model input.

## 8. Phase 8 is not reinforcement learning

Do not implement or use during Phase 8:

```text
current-policy self-play
historical-checkpoint opponents for data generation
PPO / clipped policy-ratio updates
advantage estimation
advantage filtering
KL to the behavior policy
dynamic damping
magnet-policy regularization
EMA as a self-play stabilization mechanism
learned setup selection
setup-policy training
search-generated policy targets
decision-time search
human game data
```

Those belong to later phases. Rule-generated final outcomes may supervise the value head. That is supervised outcome learning, not Phase 9 self-play RL.

## 9. Frozen synthetic population expectation

Agent 1 must read the live accepted Phase 4 opponent roster.

The expected accepted roster is:

```text
4 tier policies:
    strategic
    tactical
    basic
    random

6 unusual/stress policies:
    the exact accepted Phase 4 stress roster
```

Expected total:

```text
10 rule-based policies
100 ordered red-policy / blue-policy matchup cells
```

If the live accepted Phase 4 roster materially differs, Agent 1 must report `BLOCKED` before defining a replacement schedule.

Phase 8 does not modify these policies.

## 10. Policy-supervision rule

The synthetic corpus has two purposes: expose the model to diverse games and imitate sensible rule-agent actions. Not every rule agent is a good action teacher.

Freeze the following policy-loss eligibility unless Agent 1 discovers a concrete incompatibility in the live Phase 4 contracts:

```text
strategic decisions    policy weight 1.0
tactical decisions     policy weight 1.0
basic decisions        policy weight 0.5

random decisions       policy weight 0.0
stress decisions       policy weight 0.0
```

Random/stress positions still supervise value and belief and still shape the states in which the strategic/tactical/basic opponent must act.

Do not imitate deliberately unusual behavior merely because it exists in the corpus.

A different weighting requires review before corpus generation.

## 11. Synthetic corpus split and size

Use separate Phase 7 setup splits.

For every ordered rule-policy pair:

```text
train games / ordered pair          200
validation games / ordered pair      40
test games / ordered pair            40
```

With the expected 10-policy roster:

```text
ordered matchup cells               100

train games                      20,000
validation games                  4,000
test games                        4,000
total                            28,000
```

Every split uses domain-separated seeds.

Every game independently samples both setups from the corresponding Phase 7 split using the frozen `neutral_v1` profile.

The train split is the only split whose examples may update weights.

Validation may choose pilot/training configuration and checkpoint.

Test is structurally audited earlier but **sealed from all model-selection decisions until Agent 7**.

The frozen Phase 4 setup bank is not part of the corpus and is sealed from pilot/config selection.

## 12. Synthetic corpus game identity

Agent 1/2 must define a stable `synthetic_game_id`.

It must be a pure function of at least:

```text
corpus version
corpus split
red policy id/version
blue policy id/version
ordered-pair game ordinal
corpus master seed
```

Setup and action tie-break RNG streams must be domain-separated from one another.

Changing worker count, process partitioning, arrival order, or resume boundary must not change a logical game's setups, rule-agent seeds, actions, result, terminal reason, or metadata for the same corpus identity.

## 13. Compact storage rule

Do not materialize every observation tensor.

Store compact games using the accepted trajectory/replay machinery plus synthetic metadata sufficient to reproduce:

```text
rule policy identities
rule policy seeds
setup provenance
ordered matchup identity
corpus split
final outcome
selected policy-supervision weight by acting policy
```

Observations, legal masks, value labels, and belief labels are reconstructed from replay.

Snapshots should support efficient random access.

## 14. Deterministic per-game decision sampling

Long games must not dominate the warm start solely because they have more plies.

Freeze `warmstart_decision_sampler_v1`:

```text
maximum selected decisions per game = 64
```

For a game with `T <= 64` decisions, select all decisions.

For `T > 64`:

1. partition decision indices `[0,T)` into 64 contiguous near-equal bins;
2. choose exactly one decision from each bin using a domain-separated deterministic stream based on the game identity and bin index;
3. select without replacement;
4. sort selected indices before reconstruction.

This gives opening/middle/end coverage without using game outcome, teacher strength, future value, or model predictions to choose positions.

The final result may label a selected value target, but it must not decide whether that position is sampled.

## 15. Training-example contract

Expected version:

```text
warmstart_example_v1
```

Each selected decision reconstructs:

```text
observation              [127,10,10] float32
legal_mask               [10000] bool/uint8
acting_player             RED/BLUE metadata before conversion

policy_action_abs         trajectory action, absolute engine id
policy_action_model       perspective-normalized action id
policy_weight             from acting rule-policy contract

value_target              WIN/DRAW/LOSS from acting-player perspective

belief_target             [100], true opponent type on unresolved hidden
                          opponent squares in model frame; ignore elsewhere
belief_mask               [100] true exactly where belief is supervised

game_id
decision_index
source policy id
corpus split
```

Only `observation` is passed into the model.

The legal mask is used by the policy loss / action adapter, not as a hidden observation channel.

## 16. Target definitions

### 16.1 Policy target

For policy-supervised decisions:

```text
target = actual legal action chosen by the rule policy
```

Use masked cross entropy over legal model-frame actions.

Illegal actions are excluded from the policy normalization.

For decisions with `policy_weight == 0`, there is no policy gradient contribution. They may still train value/belief.

### 16.2 Value target

For every selected decision:

```text
final winner = acting player      -> WIN
draw                              -> DRAW
final winner = opponent           -> LOSS
```

Use categorical cross entropy over `[WIN,DRAW,LOSS]`.

Do not bootstrap from model values in Phase 8.

### 16.3 Belief target

For every opponent piece that is still unresolved to the acting player:

```text
target square = perspective-normalized current square
target class  = true piece type
```

All own pieces, empty/lake squares, and opponent pieces whose identity is already legally known are ignored.

Use hidden-only masked cross entropy.

The true type may come from privileged replay state only after the public observation is constructed.

## 17. Baselines frozen before training

Agent 1 defines and Agent 3 implements these baselines.

### Policy baseline

Uniform over legal actions:

```text
p(a) = 1 / number_of_legal_actions
```

Measure cross entropy and expected top-1 accuracy `mean(1 / legal_count)` on policy-supervised examples only.

### Value baseline

One constant W/D/L distribution fitted from **train selected examples only**.

Validation/test use that frozen train prior.

Measure categorical cross entropy, Brier score, and accuracy.

### Belief baseline

Observable remaining-inventory marginal:

```text
p(type=t) =
    unresolved_remaining_count[t] /
    total_unresolved_remaining_count
```

applied independently to each unresolved hidden opponent piece.

Measure hidden-only cross entropy and top-1 accuracy.

This is a simple marginal baseline. Phase 11 still owns full rule-consistent belief sampling and deeper belief validation.

## 18. Training loss

Each component is normalized over its own valid supervision elements before combination:

```text
L = lambda_policy * L_policy
  + lambda_value  * L_value
  + lambda_belief * L_belief
```

`L_policy` is normalized by the sum of nonzero policy weights.

`L_value` is normalized over selected decisions.

`L_belief` is normalized over supervised hidden pieces.

Do not allow the number of hidden pieces in one batch to silently multiply the belief head's global influence.

Agent 1 freezes the candidate loss-weight set before pilots.

## 19. Optimizer / training family

Phase 8 uses:

```text
model                    C1 only for meaningful training
training precision       float32 on MPS
optimizer family         AdamW
batch size               256 unless Agent 4 proves a larger fixed batch is
                         strictly preferable and Agent 5's frozen candidate
                         matrix already allows it
gradient clipping        required
learning-rate schedule   versioned, explicit
```

C0 may be used for tiny trainer unit tests only.

Do not change architecture to improve warm-start metrics.

Do not introduce mixed precision into the production warm start unless explicitly reviewed after Agent 4 evidence; float32 is the Phase 8 reference.

## 20. Pilot-budget rule

Hyperparameter search must be small, explicit, and frozen **before the first meaningful pilot update**.

Agent 1 must publish a candidate matrix containing no more than:

```text
6 configurations
```

Allowed dimensions:

```text
learning rate
loss weights
optional weight decay / schedule variant
```

Do not search architecture, setup distribution, teacher roster, teacher policy weights, test data, or Phase 4 evaluation strength.

Pilot runs must use the same model initialization seed, training decision-index universe, pilot update budget, and validation checkpoints unless the candidate parameter itself requires otherwise.

## 21. Pilot-selection metric

Freeze before training:

```text
r_policy = model policy CE / uniform-legal policy CE
r_value  = model value CE  / train-prior value CE
r_belief = model belief CE / remaining-count belief CE

selection_score = mean(r_policy, r_value, r_belief)
```

Lower is better.

Hard veto:

```text
non-finite loss/gradient/parameter
target mismatch
data split leak
checkpoint/resume failure
any component ratio > 1.05 at the final pilot checkpoint
```

Tie-break order:

1. lower selection score;
2. lower validation policy ratio;
3. higher measured training examples/s.

Do not use test metrics or game strength for pilot selection.

## 22. Canonical Phase 8 model initialization

Agent 1 must freeze one corpus master seed, one canonical C1 model initialization seed, one train-order/shuffle seed, one pilot seed namespace, and one final-run seed namespace.

Seeds must be chosen before any model result is observed.

Agent 6's production warm-start run must start from a **fresh reconstruction of the canonical random C1 initialization**, not from a pilot checkpoint.

## 23. Checkpoint/resume contract

Expected:

```text
warmstart_checkpoint_v1
```

A resumable checkpoint must include enough state to continue the exact logical training run:

```text
model architecture/config identity
model state_dict
optimizer state
scheduler state

global optimizer step
examples consumed
epoch / sampler position
best-validation checkpoint state
validation history required for early stopping

corpus version + manifest digests
training-example version
training-config version

model-init seed
shuffle/order seed
Python RNG state if used
NumPy RNG state if used
Torch CPU RNG state if used
all explicit data-sampler RNG/counters

wall-clock metadata
source revision
software versions
```

Because Phase 8 trains from a static corpus, it does not need to resume live neural self-play.

Agent 4 must prove same next batch after resume, same data identities/order, same optimizer/scheduler counters, same logical validation cadence, and a numerically equivalent parameter path before Agent 5 pilots.

## 24. Corpus crash/restart requirement

Phase 7 identified a theoretical crash window between provenance-sidecar and trajectory writes.

Phase 8's synthetic corpus generator must explicitly close this issue for the static corpus.

A corpus record is trainable only if it is **committed** after both the trajectory payload and synthetic/setup metadata exist and verify.

Use a commit journal/index or equivalent.

On resume:

```text
scan persisted game ids
scan metadata game ids
scan commit ids
reconcile
never duplicate a committed game
never expose an orphan to the dataset
rebuild or discard incomplete/uncommitted work deterministically
```

After finalization require:

```text
orphan trajectory records   0
orphan metadata records     0
duplicate committed ids     0
missing committed records   0
```

Do not change `trajectory_v1` just to accomplish this.

## 25. Held-out data discipline

### Train corpus

May update weights, fit value prior, and compute optimization statistics.

### Validation corpus

May select pilot configuration, select best checkpoint, and drive early stopping. May not update weights.

### Test corpus

Before Agent 7: structural integrity checks only; no model metrics, hyperparameter decisions, or checkpoint selection.

Agent 7 opens it once the Agent 6 checkpoint is frozen.

### Phase 4 evaluation bank

Before Agent 7: do not use neural playing strength for pilot/config/checkpoint selection.

Agent 7 uses it for the final random-baseline gate.

## 26. Phase 8 final acceptance thresholds

These thresholds are frozen before training.

### 26.1 Playing-strength gate

Using the frozen Phase 4 evaluation bank and paired color-swap semantics:

```text
opponent                       RandomPolicy / frozen Phase 4 random tier
evaluation pairs               all 1,024 setup pairs
games                          2,048

effective win rate             >= 0.950
red-side effective win rate    >= 0.900
blue-side effective win rate   >= 0.900
paired-bootstrap 95% lower
confidence bound on EWR        > 0.900

illegal moves                  0
model failures                 0
non-finite outputs             0
```

This exceeds the project plan's minimum "95% effective win rate over 1,000 games against random legal play" in sample size while retaining the same point target.

### 26.2 Improvement over initialization

Against the canonical untrained C1 checkpoint:

```text
at least 512 paired setup cases / 1,024 games
final-checkpoint EWR >= 0.700
paired-bootstrap lower bound > 0.550
```

This operationalizes the project's "clear superiority over early checkpoints" criterion.

### 26.3 Policy learning

On sealed synthetic test examples:

```text
policy CE <= 0.90 * uniform-legal CE
policy top-1 accuracy > uniform-legal expected top-1 accuracy
```

### 26.4 Value learning

On sealed synthetic test examples:

```text
value CE <= 0.98 * frozen train-prior CE
value Brier score < train-prior Brier score
```

### 26.5 Belief learning

On hidden-only sealed test targets:

```text
belief CE <= 0.98 * remaining-count-prior CE
belief top-1 accuracy > remaining-count-prior top-1 accuracy
```

These are deliberately modest "nontrivial prediction" gates. Full belief-system validation remains Phase 11.

### 26.6 Stability / non-collapse

On sealed test policy states:

```text
finite logits on 100% of examples
fraction with legal max probability > 0.999 < 0.95
```

Report normalized legal-action entropy distribution.

This catches catastrophic deterministic collapse without requiring a strong policy to remain artificially high-entropy.

## 27. Statistical reporting

Policy/value/belief test confidence intervals must be bootstrapped by **game**, not by individual decision/piece, so correlated positions from one game are not treated as independent observations.

Playing-strength evaluation uses the already accepted paired bootstrap semantics.

Report point estimate, 95% confidence interval, number of games, number of decisions, and number of hidden-piece targets.

## 28. Development training budget

Phase 8 is not the final 168-hour run.

Agent 1/5 may freeze a practical development budget, but:

```text
pilot candidates       <= 6
pilot updates/config   <= 5,000
canonical final run    <= 25,000 optimizer steps
```

unless review explicitly authorizes more.

The goal is to establish a good warm start and a trustworthy training system, not maximize Phase 8 Elo.

## 29. Evaluation reporting beyond hard gates

Agent 7 should also report, without turning them into Phase 8 hard gates unless Agent 1 explicitly froze one:

```text
EWR vs Basic
EWR vs Tactical
EWR vs Strategic
performance by color
performance by Phase 4 setup bank subgroup if available
performance by synthetic test setup family
terminal-reason distribution
game-length distribution
policy entropy
value confusion matrix
belief CE / accuracy by hidden piece type
belief CE by game-progress bucket
```

These diagnostics guide Phase 9.

Do not reopen the Phase 8 configuration after seeing them.

## 30. Common correctness rules

Every agent must:

- read the real repository and all prerequisite artifacts;
- run the full suite before edits and record exact totals;
- use `.venv/bin/python` or the repository's accepted equivalent environment;
- preserve all earlier tests;
- add a regression for every discovered bug;
- fail loudly rather than substitute data or actions;
- keep all RNG explicit and domain-separated;
- preserve the frozen action-frame conversion;
- use the engine as legality/replay authority;
- preserve observer safety;
- preserve Phase 4 bank identity;
- preserve Phase 7 library and split identity;
- record every meaningful run seed and command;
- write machine-readable artifacts before final artifact-gated tests;
- stop at the end of the assigned agent.

## 31. Common stop conditions

Report `BLOCKED` instead of inventing semantics if:

- any prerequisite agent is not PASS;
- accepted upstream versions/digests disagree with live source;
- the rule-based roster differs materially from the accepted Phase 4 contract;
- a target cannot be reconstructed without modifying engine semantics;
- policy/value/belief labels cannot be defined without leakage;
- action normalization is ambiguous;
- a static corpus cannot resume/reconcile without exposing corrupt records;
- trainer checkpoint/resume cannot restore the logical data/update state;
- C1 cannot perform finite optimizer updates on MPS;
- validation/test split isolation cannot be enforced;
- achieving an acceptance threshold would require using test data for tuning;
- Phase 9 RL machinery would be required to make Phase 8 function.

Ordinary implementation bugs should be fixed with regressions.

A model failing a predeclared learning threshold is `FAIL` or a reason for explicit review—not authorization to move the threshold.

## 32. Common machine-readable metadata

Every Agent 1–7 primary JSON should record at least:

```text
phase
agent
status
timestamp
source_revision
working_tree_state

platform
python_version
torch_version
mps_built
mps_available

prerequisite_versions
prerequisite_digests
tests_before
tests_after
commands
durations
seeds

files_created
files_modified
completion_gates
problems
deviations
```

Where relevant also record corpus version/digests/counts, training-example/trainer/checkpoint/train-config versions, C1 identity, optimizer/scheduler/loss weights, updates/examples/throughput/memory, validation metrics, test metrics, and evaluation results.

## 33. Global Phase 8 acceptance

Agent 7 may recommend `PASS` only if all are true:

```text
Agents 1-6                                     PASS

upstream engine/rules/observation              unchanged
C1 architecture/config                         unchanged
Phase 4 bank                                   unchanged
Phase 7 library/sampler                        unchanged

synthetic corpus                               finalized and deterministic
train games                                    exact schedule
validation games                               exact schedule
test games                                     exact schedule
ordered rule-policy pair coverage              complete
corpus split leakage                           0
corpus replay mismatches                       0
orphan committed records                       0
duplicate committed game ids                   0

training-example reconstruction mismatches     0
policy target mismatches                       0
value target mismatches                        0
belief target mismatches                       0
input/target hidden-information leaks          0
absolute/model action-frame mismatches         0

C1 finite optimizer path                       PASS
checkpoint/resume                              PASS
data cursor/order resume                       PASS

pilot configs                                  <= 6
pilot selection used train+validation only     YES
test used for pilot/config selection           NO
Phase 4 strength used for selection            NO

canonical final run began from fresh C1 init   YES
best checkpoint selected by validation only    YES

random EWR                                     >= 0.950 over 2,048 games
red EWR                                        >= 0.900
blue EWR                                       >= 0.900
paired 95% EWR lower bound                     > 0.900

final vs canonical initialization EWR          >= 0.700 over >=1,024 games

test policy CE ratio                           <= 0.90
test value CE ratio                            <= 0.98
test value Brier                               better than baseline
test belief CE ratio                           <= 0.98
test belief top-1                              better than baseline

non-finite held-out logits                     0
catastrophic policy collapse gate              PASS

meaningful Phase 9 self-play/RL occurred       NO
learned setup selection occurred               NO
decision-time search training occurred         NO

full repository suite                          GREEN
```

## 34. Phase 8 handoff to Phase 9

After formal acceptance, Phase 9 receives:

```text
frozen warm-start checkpoint
checkpoint digest
warmstart_train_config_v1
warmstart_checkpoint_v1
warmstart_example_v1
synthetic_warmstart_corpus_v1 manifest/digests

training and validation curves
sealed-test results
Phase 4 random/basic/tactical/strategic evaluation report

known failure modes
throughput/memory numbers
resume semantics
corpus regeneration instructions
```

Phase 9 may initialize the current policy from the accepted Phase 8 warm-start checkpoint.

Phase 9 must not silently alter Phase 8 corpus/test evidence.

Stop Phase 8 after Agent 7's handoff.
