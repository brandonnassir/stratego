# Phase 9 — Agent 4
# RL Targets, Advantages, and Anti-Leak Audit

## Mission

Implement and independently verify all Phase 9 trainable-example semantics:

- learner-side extraction;
- same-player temporal sequences;
- scalar lambda advantages;
- WDL lambda targets;
- advantage filtering;
- PPO eligibility;
- belief targets;
- behavior-policy consistency;
- information-security boundaries.

Do not perform meaningful RL training.

## Prerequisites

Require Agents 1–3 `PASS` and formal acceptance.

Verify rollout contract/store digests and behavior reproduction evidence.

### Mandatory corpus resolver check

Resolve the accepted Phase 8 corpus via `synthetic_corpus.default_corpus_root()` and require all accepted digests. No hard-coded path in target/dataset code.

## Same-player extraction

For every trainable side:

- iterate only that player's learner-controlled decisions;
- preserve game order;
- do not insert opponent decisions as learner steps;
- terminal may occur before that player's next turn;
- maintain one fixed player perspective for the entire sequence.

Exhaustively test Red/Blue and `learner_control = red|blue|both`.

## Scalar behavior value

From stored behavior WDL:

\[
v_t=P_t(W)-P_t(L)
\]

Require WDL probabilities finite and normalized according to Agent 1's contract.

## Advantage

Use:

```text
gamma      1.0
lambda_A   0.5
```

For continuation to that player's next learner decision:

\[
\delta_t=v_{t+1}-v_t
\]

If terminal before that player's next learner decision:

\[
\delta_t=z-v_t
\]

Then backwards:

\[
A_t=\delta_t+0.5A_{t+1}
\]

No sign flip from opponent turns because the sequence remains in one player's perspective.

Implement independent reference arithmetic in tests, not by calling the production helper.

## WDL lambda target

Use:

```text
lambda_V = 0.8
```

Terminal:

\[
Y_t=Z
\]

Otherwise:

\[
Y_t=0.2P_{t+1}+0.8Y_{t+1}
\]

Require each target:

- finite;
- nonnegative within numeric tolerance;
- sums to 1 within tolerance;
- expressed from the learner's fixed perspective.

## Advantage filter

Per sealed rollout iteration:

\[
\tau=\max(Q_{0.75}(|A|),0.01)
\]

Policy eligibility iff:

\[
|A|\ge\tau
\]

Then standardize eligible advantages using only the eligible PPO subset.

Handle zero-variance edge cases explicitly and freeze behavior.

Value/belief eligibility is not filtered.

## Belief target

Continue the accepted hidden-only Phase 8 belief target semantics on every learner decision:

- opponent unresolved hidden pieces only;
- privileged truth used only after observation creation;
- model-frame square mapping;
- ignore own/revealed/empty/lake;
- accepted 12-type semantics.

No belief label enters model input.

## Dataset/example contract

Create a Phase 9 RL example/batch contract containing only fields needed downstream.

Targets/metadata may include:

```text
observation
legal mask
sampled action
behavior action probability/logprob
behavior legal distribution if frozen
advantage
standardized advantage
PPO eligibility
WDL target
belief target/mask
game id
decision index
learner side
behavior checkpoint id
rollout id
```

Only observation enters the neural backbone input; legality is used only for masking.

## Exhaustive target audit

On at least one substantial sealed rollout:

- audit every learner decision;
- independently recompute learner designation;
- independently recompute final-perspective outcome;
- independently recompute same-player next-state links;
- independently recompute all advantages;
- independently recompute all WDL lambda targets;
- independently recompute filter threshold and eligibility;
- independently rebuild belief labels;
- verify behavior quantity against stored collection record.

Zero mismatches.

## Anti-leak trials

Run at least:

```text
25,000 valid hidden-identity permutation trials
```

Use unresolved opponent identity permutations.

Require identical:

```text
observation bytes
legal actions
model action mapping
behavior-model input
learner designation
public/behavior-derived PPO inputs
belief mask
```

Privileged truth/belief labels should change exactly when the hidden assignment changes.

Add positive controls proving the audit detects:

- privileged identity planted in observation;
- privileged metadata attached to model input;
- wrong action frame;
- wrong value perspective;
- wrong learner-control side.

## Behavior consistency

Independently re-check at least 100,000 learner decisions against the exact frozen behavior snapshot and Agent 1 tolerance.

This should not simply call Agent 3's acceptance function.

## Artifacts

Create:

```text
reports/phase_9_data/agent_04_target_audit.json
reports/phase_9_data/agent_04_antileak.json
reports/phase_9_data/agent_04_example_contract.json
```

## Completion gates

Require:

```text
agents1_3_pass
corpus_resolver_verified
corpus_digests_match
same_player_sequence_audit_pass
red_blue_perspective_audit_pass
advantages_exhaustively_match
wdl_targets_exhaustively_match
advantage_filter_exact
value_target_simplex_failures_zero
belief_target_mismatches_zero
behavior_reproduction_ge_100k
behavior_reproduction_mismatches_zero
hidden_permutation_trials_ge_25000
model_input_leak_mismatches_zero
positive_controls_fire
learner_control_mismatches_zero
no_meaningful_rl_training
full_suite_green
```

## Forbidden

Do not:

- tune lambda values;
- change filter quantile/minimum;
- run a pilot;
- train PPO meaningfully;
- select a checkpoint;
- open final-test bank;
- change observer representation;
- leak privileged truth into action/value policy input.

## Handoff to Agent 5

Provide:

- deterministic rollout-to-example iterator;
- Phase 9 train order/cursor;
- PPO eligibility mask;
- stored behavior quantity;
- standardized advantages;
- WDL targets;
- belief targets;
- exact target/anti-leak evidence;
- resumable minibatch cursor semantics.

Agent 5 implements optimization only.
