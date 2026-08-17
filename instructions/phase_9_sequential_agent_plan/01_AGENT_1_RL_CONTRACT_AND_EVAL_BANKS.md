# Phase 9 — Agent 1
# RL Contract, Evaluation Banks, and Acceptance Freeze

## Mission

Freeze every Phase 9 learning and evaluation semantic **before any Phase 9 optimizer step or trainable self-play rollout is generated**.

You are the contract agent. Later agents implement your frozen specification rather than make new RL-design choices.

Read `00_PHASE_9_SEQUENCE_AND_COMMON_CONTRACT.md` completely before doing anything.

## Prerequisites

Require:

```text
Phase 8 status     PASS — formally accepted
Agents 1–7         PASS
accepted C1        863,959 parameters
accepted checkpoint SHA
f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca
```

Verify the accepted Phase 8 checkpoint and canonical-untrained checkpoint by SHA-256 and normal load path.

Verify C1 config digest:

```text
31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d
```

Verify the frozen rules, observation, action frame, setup sampler/library, trajectory foundation, and Phase 4 policy roster.

### Mandatory corpus resolver check

Resolve the accepted Phase 8 corpus only through:

```text
synthetic_corpus.default_corpus_root()
```

Expected current result:

```text
/Users/brandonwashington/Dev/Github/stratego/gpt_agent/data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1
```

Require all three accepted corpus digests. Do not hard-code this path into reusable implementation. A mismatch is `BLOCKED`.

## Required decisions to freeze

Freeze, version, serialize, and regression-test:

```text
phase9_rl_contract_v1
phase9_population_v1
phase9_rollout_schedule_v1
phase9_rollout_store_v1
phase9_advantage_v1
phase9_train_order_v1
phase9_checkpoint_v1
phase9_eval_bank_v1
phase9_acceptance_v1
```

Do not leave any learning-critical behavior implicit.

Freeze exactly:

- all Phase 9 seeds from the common contract;
- population proportions;
- exact canonical and pilot game counts;
- rule-bucket subdivision;
- historical archive cadence/window;
- learner-control semantics;
- behavior-policy temperature and legal softmax semantics;
- behavior-probability/log-prob storage representation;
- storage dtype/precision and numerical verification tolerance;
- same-player temporal sequence definition;
- `gamma=1.0`;
- `lambda_A=0.5`;
- `lambda_V=0.8`;
- advantage filtering and standardization;
- PPO clip `0.20`;
- behavior-KL direction and target;
- adaptive KL beta rule and bounds;
- entropy schedule;
- value/belief weights;
- minibatch size and optimizer epochs;
- pilot matrix;
- pilot hard vetoes;
- validation score and tie-break;
- validation bank;
- final-test bank;
- canonical 60-iteration budget;
- all final hard gates;
- bootstrap unit/seeds and exact CI calculation;
- sealing rules.

## Behavior-storage decision

Inspect the actual repository representation before freezing this.

The semantic requirement is:

> For every trainable neural decision, downstream code must be able to verify the realized action probability under the exact immutable behavior snapshot used to generate that action.

If `trajectory_v1` can represent this faithfully without changing its meaning, reuse it.

If not, create a versioned Phase 9 wrapper/sidecar contract. Do **not** silently reinterpret fields frozen by earlier phases.

Record:

- stored quantity: full legal distribution vs action log-prob vs equivalent;
- dtype;
- normalization rule;
- reconstruction/verification rule;
- maximum allowed numeric mismatch;
- how opponent rule-policy decisions are represented when they are not PPO-trainable.

## Evaluation banks

Create deterministic, hashed manifests before training.

### Validation

```text
phase9_validation_bank_v1

128 paired setup cases
Phase 7 validation split
8 paired cases per each of 16 setup families
color_swap_same_board
```

Core opponents:

```text
Phase8 anchor
Random
Basic
Tactical
Strategic
```

No outcome-based case selection.

### Final test

```text
phase9_test_bank_v1

512 paired setup cases
Phase 7 test split
32 paired cases per each of 16 setup families
color_swap_same_board
```

Structurally audit only. No neural checkpoint may play a final-test case before Agent 8.

Freeze a smaller deterministic report-only stress schedule.

## Baseline anchor evaluation

Before the first Phase 9 update, it is permissible to evaluate the already frozen Phase 8 anchor on the Phase 9 **validation** bank because that bank is for model selection.

Record anchor validation EWRs for:

```text
Random
Basic
Tactical
Strategic
```

Do not open the Phase 9 final-test bank with a neural model.

For final paired improvement against Tactical/Strategic, freeze the procedure by which Agent 8 will evaluate the Phase 8 anchor on the same final cases after the final bank is legitimately opened.

## Sealing

Implement/test explicit guards.

Before Agent 8:

```text
phase9_test_bank:
    structural_audit            ALLOWED
    neural_model_inference      REFUSED
    model_metric                REFUSED
    checkpoint_selection        REFUSED
    hyperparameter_selection    REFUSED
```

No test metric may influence Agents 1–7.

## Pilot and canonical budgets

Freeze exactly the six pilot candidates from the common contract.

Pilot:

```text
8 RL iterations
1,024 games / iteration
2 optimizer epochs
```

Canonical:

```text
60 RL iterations
2,048 games / iteration
2 optimizer epochs
validation every 5 iterations
archive every 5 iterations
12-hour operational ceiling
```

No alternate budget may be invented later unless you explicitly freeze it now.

## Acceptance artifacts

Create:

```text
reports/phase_9_data/agent_01_rl_contract.json
reports/phase_9_data/agent_01_acceptance.json
reports/phase_9_data/agent_01_validation_bank.json
reports/phase_9_data/agent_01_test_bank.json
```

Include exact digests for every serialized contract/bank.

## Required tests

At minimum:

- contract round-trip;
- seed domain separation;
- exact game-count arithmetic;
- population proportions;
- rule-subdivision arithmetic;
- learner-control semantics;
- advantage/filter formulas on hand-computable sequences;
- WDL lambda target on hand-computable sequences;
- pilot matrix exactly six;
- validation-score arithmetic;
- final-gate arithmetic;
- test-bank access refusal;
- validation-bank access allowance;
- bank family balance;
- color-pairing exactness;
- no final-test model inference in this agent.

## Completion gates

Report PASS only if all are true:

```text
phase8_identity_verified
corpus_resolver_verified
corpus_digests_match
rl_contract_frozen
population_contract_frozen
rollout_schedule_frozen
behavior_storage_semantics_frozen
advantage_contract_frozen
checkpoint_contract_frozen
pilot_matrix_exactly_six
validation_score_frozen
validation_bank_frozen_and_hashed
test_bank_frozen_and_hashed
test_bank_neural_access_zero
final_gates_frozen
no_phase9_optimizer_steps
no_trainable_phase9_rollouts
full_suite_green
```

## Forbidden

Do not:

- implement the production scheduler;
- collect the production Phase 9 rollout corpus;
- run PPO;
- run a pilot;
- tune from outcomes;
- open the Phase 9 final-test bank with a neural model;
- touch Phase 8 checkpoint weights;
- modify engine/setup/model contracts.

## Handoff to Agent 2

Provide exact versions/digests plus:

- population bucket schedule;
- deterministic game-ID specification;
- opponent-ID specification;
- color-balance rule;
- historical archive identities and active-window rule;
- setup assignment rule;
- validation/test bank manifests;
- all seed derivations;
- learner-control semantics.

Agent 2 makes no new learning-design decision.
