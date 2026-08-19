# Optional Phase 10B — Setup-Conditioned Self-Play Fine-Tuning Agent Plan

## Status and role

Phase 10B is an **optional follow-up experiment** that may run in parallel with the main roadmap.

It is **not** part of the formally closed Phase 10 acceptance result, does **not** reopen Phase 10, and must **not** delay Phase 11 or Phase 12.

The main project proceeds as:

```text
Phase 10 — CLOSED
Phase 11 — proceeds now
Phase 12 — follows Phase 11 if authorized
```

Phase 10B runs separately:

```text
Frozen Phase 9 checkpoint
        +
Frozen Phase 10 P10-D selector
        ↓
P10-D-conditioned self-play
        ↓
bounded Phase 9-style PPO/KL fine-tuning
        ↓
candidate Phase 10B checkpoint
        ↓
independent comparison against frozen controls
```

The Phase 10B result is advisory. When it finishes, the reviewing chat may decide whether the checkpoint is worth revisiting after the initial Phase 11/12 development results.

No Phase 10B result may retroactively change the accepted Phase 10 classification or artifacts.

---

# 1. Scientific question

Phase 10 established that the learned P10-D setup selector was **non-inferior** to `neutral_v1`, but the move policy itself was never trained while operating under the selected setup distribution.

Phase 10B asks:

> If the accepted Phase 9 move policy is fine-tuned in self-play while both sides use the frozen P10-D setup selector, does the move policy adapt beneficially to the setup distribution without losing the broad strength, safety, diversity, or stability established by Phase 9 and Phase 10?

The experiment is specifically testing **setup-conditioned policy adaptation**.

It is not testing a new selector, a new setup generator, or a new belief architecture.

---

# 2. Frozen upstream inputs

The agent must verify all identities from live bytes before any Phase 10B rollout is generated.

## Phase 9 move model

Use the accepted Phase 9 checkpoint as the **only initialization**:

```text
checkpoint
checkpoints/phase9/selfplay_c1_v1.pt

SHA-256
dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea

model-state digest
f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd

parameters
863,959
```

The accepted Phase 9 checkpoint remains immutable and is the permanent rollback/reference anchor.

## Frozen setup selector

Use the formally accepted Phase 10 selector:

```text
candidate
P10-D

utility
model_T

temperature
0.75

mixture
0.35 neutral_v1 / 0.65 learned

selector config SHA-256
6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668
```

The selector, utility coefficients, trait scaler, mixture, temperature, Phase 7 library, reflection behavior, and perturbation behavior are immutable.

## Phase 10 utility identities

```text
model_T coefficient digest
d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc

trait scaler digest
fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9
```

## Phase 10 production identity

```text
filled production-system digest
615cc3c3a4fab6e4400e20a5a93b13a08c43ab6c3ca63828c6a64742e98175d2
```

Phase 10B must consume these artifacts; it must never rewrite them.

---

# 3. Fixed controls

Three systems are relevant throughout Phase 10B.

## Control N — Phase 9 + neutral

```text
move policy
accepted frozen Phase 9 checkpoint

setup source
neutral_v1
```

This is the original broad reference system.

## Control D — Phase 9 + P10-D

```text
move policy
accepted frozen Phase 9 checkpoint

setup source
frozen P10-D
```

This isolates what the selector alone contributes.

## Candidate B — Phase 10B + P10-D

```text
move policy
Phase 10B fine-tuned checkpoint

setup source
the same frozen P10-D
```

This isolates the effect of policy adaptation **after** setup conditioning.

No other control or candidate may be introduced after outcomes are observed.

---

# 4. Absolute prohibitions

The Phase 10B agent may not:

1. alter the accepted Phase 9 checkpoint;
2. alter P10-D;
3. refit the Phase 10 utility model;
4. change P10-D temperature;
5. change the 0.35 / 0.65 mixture;
6. change `neutral_v1`;
7. mutate the Phase 7 setup library;
8. change the 127-channel observation design;
9. change the belief-head architecture;
10. introduce search into training;
11. use Phase 11 or Phase 12 test evidence for training;
12. use Phase 10 final-test outcomes as training data;
13. change Phase 9 PPO/KL mechanics except where this plan explicitly freezes a Phase 10B schedule;
14. run a 168-hour final training budget;
15. block, pause, or alter Phase 11 execution.

Phase 10B is a bounded side experiment only.

---

# 5. Training method

Use the **already validated Phase 9 PPO/KL machinery**.

The implementation should reuse the accepted Phase 9 training path rather than inventing a new RL trainer.

The frozen Phase 9 mechanics remain:

```text
policy objective
PPO clipped objective

PPO clip
0.20

behavior KL target
0.015

hard KL veto
epoch mean KL > 0.08

hard clip-fraction veto
clip fraction > 0.75

gradient clip
1.0

optimizer
accepted Phase 9 AdamW configuration

weight decay
0.01

training dtype/backend
accepted Phase 9 float32 path

training batch
512

epochs per iteration
2

replay
none

advantage lambda
0.5

WDL target lambda
0.8

advantage filter
tau = max(Q75(|A|), 0.01)

selected advantages
standardized

value/belief supervision
all learner decisions
```

If a current implementation detail differs from historical report prose, the live accepted Phase 9 contract/code is authoritative.

Do not silently reconstruct PPO from this document if the accepted Phase 9 implementation exists.

---

# 6. Setup-conditioned self-play rule

Every Phase 10B training game must use:

```text
Red setup source
P10-D

Blue setup source
P10-D
```

Both setup draws use the accepted selector independently through their normal color-specific distributions and frozen seeds.

The selector sees only legal own-side inputs.

No training game may choose a setup based on the opponent's hidden setup, hidden rank truth, outcome prediction, current checkpoint strength, or matchup identity.

P10-D is a fixed environment distribution for this experiment.

---

# 7. Opponent population

The experiment is deliberately centered on **P10-D-conditioned current-policy self-play**, but should preserve some stabilizing opponent diversity so the learner does not overfit to a single current-current loop.

Use this fixed population over games:

```text
current Phase 10B learner          60%
accepted Phase 9 anchor            20%
Phase 9 historical archive         10%
rule / stress opponents            10%
```

Within the 10% rule/stress bucket:

```text
Strategic      30%
Tactical       25%
Basic          10%
information_miser 10%
scout_rush     10%
miner_rush     10%
Random          5%
```

All games still use P10-D for the learner's own setup.

For opponents whose setup source is meaningful, use P10-D as well unless a frozen opponent implementation requires its own setup source. Any exception must be recorded, not chosen from outcomes.

### Historical archive

Use the accepted Phase 9 historical archive exactly as frozen.

Do not build a new candidate league before training begins.

A newly created Phase 10B history checkpoint may enter only the **current learner history** under the archive rule frozen below.

---

# 8. Phase 10B history policy

Archive the current Phase 10B model every 5 completed training iterations.

Active Phase 10B history:

```text
accepted Phase 9 anchor
+
up to 4 most recent eligible Phase 10B archives
```

Sample uniformly within the Phase 10B-history portion.

Do not use performance-based archive weighting.

The accepted Phase 9 anchor is never evicted.

---

# 9. Training budget

Phase 10B should be large enough to test adaptation but small enough to remain a side experiment.

## Canonical budget

```text
maximum iterations
30

games per iteration
2,048

maximum training games
61,440

maximum optimizer epochs
60

wall-clock ceiling
12 hours
```

The run ends at the earliest of:

```text
30 completed iterations
12 wall-clock hours
hard safety stop
```

Do not extend the budget because of weak results.

Do not restart from scratch after looking at outcomes to obtain a better run.

Crash/resume is permitted only from the latest committed Phase 10B training checkpoint and must not extend the 12-hour wall-clock experiment ceiling unless the reviewing chat explicitly authorizes an operational amendment **before** resumed training continues.

---

# 10. Learning-rate and regularization schedule

Start from the accepted Phase 9 training schedule but use a deliberately conservative adaptation scale.

Freeze before training:

```text
initial learning rate
0.25 × the accepted Phase 9 canonical starting LR

final learning rate
0.10 × the accepted Phase 9 canonical starting LR

schedule
linear decay across 30 iterations
```

The intent is to adapt rather than relearn.

Entropy coefficient:

```text
iteration 1
0.0010

iteration 30
0.0005

schedule
linear
```

Behavior-KL control remains the accepted Phase 9 controller, including its hard veto thresholds.

No post-hoc LR reduction or increase is allowed.

---

# 11. Value and belief heads

Phase 10B fine-tunes **C1 through the accepted shared Phase 9 training objective**, including policy, value, and belief losses exactly as Phase 9 does.

However:

- do not redesign the belief loss;
- do not add a Phase 11 calibration objective;
- do not train from Phase 11 data;
- do not use search targets;
- do not add auxiliary setup-prediction losses.

The scientific object is the same C1 model adapted under P10-D-conditioned self-play.

---

# 12. Seeds and deterministic identities

Freeze root seeds before the first rollout.

Recommended:

```text
master
20260819021

rollout schedule
20260819022

opponent selection
20260819023

setup selection
20260819024

training minibatch/order
20260819025

validation schedule
20260819026

validation bootstrap
20260819027

final bootstrap
20260819028
```

All derived randomness must use domain-separated deterministic hashing.

Required domains include at least:

```text
rollout_game
opponent_bucket
opponent_identity
red_setup
blue_setup
action_sampling
training_order
archive_selection
validation_case
bootstrap
```

No seed may depend on worker count, process id, path, wall clock, or arrival order.

---

# 13. Checkpoint identities

Use a separate Phase 10B namespace.

Recommended:

```text
checkpoints/phase10b/
```

Never overwrite:

```text
checkpoints/phase9/selfplay_c1_v1.pt
```

Canonical Phase 10B candidate checkpoint:

```text
checkpoints/phase10b/setup_conditioned_c1_v1.pt
```

Each committed checkpoint must contain or bind:

- parent Phase 9 SHA/state digest;
- P10-D config digest;
- Phase 10 utility/scaler digests;
- Phase 10B contract digest;
- iteration;
- optimizer step;
- RNG / schedule identity;
- active history identities;
- model-state digest;
- file SHA-256.

---

# 14. Validation schedule

Evaluate every 5 completed training iterations:

```text
iterations
5, 10, 15, 20, 25, 30
```

Validation exists for **checkpoint selection only** within the predeclared Phase 10B experiment.

Do not use validation to change:
- LR;
- population mix;
- setup selector;
- PPO thresholds;
- number of iterations;
- entropy schedule.

Select the winning Phase 10B checkpoint using the frozen score below.

---

# 15. Validation matchups

Evaluate each candidate checkpoint against fixed controls and opponents.

Required validation comparisons:

### Direct setup-conditioned adaptation

```text
Phase 10B candidate + P10-D
vs
Phase 9 + P10-D
```

This is the primary comparison.

### Broad rollback guard

```text
Phase 10B candidate + neutral_v1
vs
Phase 9 + neutral_v1
```

This measures whether policy adaptation remains useful when the accepted neutral setup source is restored.

### External opponents

Under P10-D setup source for the candidate:

```text
Strategic
Tactical
Phase 8 anchor
Random
Basic
```

Also evaluate against the accepted Phase 9 move policy under P10-D.

---

# 16. Validation bank

Freeze one bounded validation bank before training.

Recommended:

```text
256 logical paired cases per matchup
2 games per case
color-swapped
```

Use identical logical cases when comparing:

```text
Phase 10B + P10-D
vs
Phase 9 + P10-D
```

and when comparing neutral controls.

Bootstrap unit = logical case.

No test bank is opened during training.

---

# 17. Validation score

Define:

```text
Δ_D
EWR(10B+P10-D vs Phase9+P10-D) - 0.5

Δ_N
EWR(10B+neutral vs Phase9+neutral) - 0.5

Δ_S
EWR(10B+P10-D vs Strategic)
-
EWR(Phase9+P10-D vs Strategic)

Δ_T
same against Tactical

Δ_P8
same against Phase 8 anchor
```

Frozen validation score:

```text
S10B =
0.40 * Δ_D
+
0.20 * Δ_N
+
0.15 * Δ_S
+
0.15 * Δ_T
+
0.10 * Δ_P8
```

Random and Basic are guardrails, not score terms.

Tie-break order:

1. higher `S10B`
2. higher `Δ_D`
3. higher `Δ_N`
4. higher Strategic delta
5. lower behavior KL
6. earlier iteration

This explicitly prefers a smaller adaptation if strength is tied.

---

# 18. Validation eligibility

A checkpoint is eligible only if:

```text
Random EWR >= 0.95
Basic EWR >= 0.80

Phase9+neutral rollback comparison
EWR >= 0.48

all Phase 9 PPO hard-safety limits respected
no checkpoint corruption
no illegal-action / nonfinite / optimizer-state errors
```

If no checkpoint is eligible:

```text
Phase 10B result
FAIL
```

and no final Phase 10B candidate is promoted.

The frozen Phase 9 model remains the only accepted move model.

---

# 19. Final candidate selection

After iteration 30 or earlier bounded stop:

1. consider only scheduled validation checkpoints;
2. filter by eligibility;
3. rank by frozen `S10B` and tie-break;
4. copy the unique winning checkpoint to the canonical Phase 10B candidate path;
5. freeze its SHA/state/iteration identity;
6. do not train further.

The selected candidate is still **experimental**, not production.

---

# 20. Sealed final evaluation

After candidate selection, run a separate first-and-only Phase 10B final evaluation.

Recommended:

```text
512 logical paired cases per matchup
2 games per case
```

The final evaluation must include:

1. `10B + P10-D` vs `Phase9 + P10-D`
2. `10B + neutral` vs `Phase9 + neutral`
3. Strategic
4. Tactical
5. Phase 8 anchor
6. Random
7. Basic
8. accepted Phase 9 under P10-D as an external fixed reference where needed for seat-balanced comparison

All pairwise deltas use the same logical cases wherever applicable.

Use 10,000 paired bootstrap replicates, 95%, logical-case resampling.

---

# 21. Final hard gates

## Gate A — P10-D direct adaptation

Primary gate:

```text
EWR(10B+P10-D vs Phase9+P10-D) >= 0.52
paired 95% lower bound > 0.50
```

This requires evidence that the fine-tuned policy actually improved under the setup distribution it trained on.

## Gate B — neutral rollback guard

```text
EWR(10B+neutral vs Phase9+neutral) >= 0.49
paired 95% lower bound > 0.47
```

The adapted move policy may specialize somewhat, but cannot materially break under the old neutral setup source.

## Gate C — strong-opponent composite

Define:

```text
Δ_L =
0.45 * Δ_Strategic
+
0.35 * Δ_Tactical
+
0.20 * Δ_Phase8
```

Require:

```text
point estimate >= 0.00
95% lower bound > -0.02
```

## Gate D — individual regression guards

For each:

```text
Strategic
Tactical
Phase 8 anchor
```

require paired delta lower bound:

```text
> -0.03
```

## Gate E — easy-opponent guards

Require:

```text
Random overall >= 0.95
Random Red >= 0.90
Random Blue >= 0.90
Basic >= 0.80
```

and paired learned-minus-Phase9 lower bound:

```text
> -0.03
```

for Random and Basic.

## Gate F — PPO/training safety

Require:

```text
hard KL veto violations       0
hard clip-fraction violations 0
nonfinite losses              0
nonfinite gradients           0
optimizer corruption          0
illegal training actions      0
```

Any triggered hard veto must abort the affected training update exactly as in Phase 9.

## Gate G — belief preservation / no catastrophic auxiliary regression

Because the shared belief head is trained jointly, compare the selected Phase 10B checkpoint against Phase 9 on the accepted belief benchmark or a fresh frozen diagnostic bank.

Require:

```text
belief CE ratio
CE_10B / CE_Phase9 <= 1.05
```

and:

```text
top-1 degradation <= 0.02 absolute
```

This is a preservation gate only.

Do not introduce Phase 11 final-test evidence here.

## Gate H — upstream artifact preservation

Require byte-identical:

```text
accepted Phase 9 checkpoint
P10-D selector config
Phase 10 utility
Phase 10 scaler
Phase 7 library
```

Only the Phase 10B checkpoint and Phase 10B-specific logs/artifacts may be new.

---

# 22. Final classification

Exactly one of:

## `PASS-CANDIDATE`

All hard Gates A-H pass.

This means the Phase 10B checkpoint is a credible **optional replacement candidate** for later reconsideration.

It does **not** automatically replace Phase 9 in the main roadmap.

## `FAIL`

The experiment ran correctly but one or more hard gates failed.

The Phase 9 checkpoint remains the accepted move model.

## `BLOCKED`

The experiment's integrity, provenance, or prerequisite identities could not be established.

---

# 23. Promotion rule

Even if Phase 10B returns `PASS-CANDIDATE`:

- do not alter Phase 11 artifacts;
- do not rerun Phase 11 automatically;
- do not replace the Phase 9 checkpoint in Phase 12 automatically;
- do not mutate the formally closed Phase 10 system.

Instead return the Phase 10B result to the reviewing chat.

The reviewing chat may later choose among:

```text
A. keep Phase 9 as the main move model
B. run a bounded cross-check of 10B under Phase 11/12
C. create a future integration phase comparing Phase 9 vs 10B
```

No automatic promotion.

---

# 24. Storage and operational policy

Prefer the external volume for:

- rollout shards;
- trajectory stores;
- Phase 10B archives;
- replay/evaluation records.

Use internal storage for:

- active checkpoint;
- small manifests;
- reports;
- hot logs.

Physical path is diagnostic, never logical identity.

Crash-safe stores should use the already validated commit/recovery semantics from earlier phases where practical.

---

# 25. Required artifacts

Recommended:

```text
reports/phase_10b_implementation_report.md

reports/phase_10b_data/
  agent_10b_contract.json
  agent_10b_training_manifest.json
  agent_10b_iteration_metrics.csv
  agent_10b_validation_results.csv
  agent_10b_selected_checkpoint.json
  agent_10b_final_results.csv
  agent_10b_belief_preservation.json
  agent_10b_acceptance.json
```

Training checkpoint namespace:

```text
checkpoints/phase10b/
```

Do not write into the accepted Phase 9 or Phase 10 artifact namespaces.

---

# 26. Required report contents

The final report must include:

- starting repository revision;
- final repository revision;
- all frozen upstream identities;
- Phase 10B contract digest;
- root seeds and derived domains;
- exact training budget consumed;
- completed games;
- optimizer updates;
- KL statistics;
- clip-fraction statistics;
- advantage retention;
- archive/history identities;
- validation result at every scheduled checkpoint;
- selected checkpoint identity;
- final test results and paired confidence intervals;
- belief-preservation result;
- all hard gate rows;
- final classification;
- explicit statement that Phase 11 was not blocked or modified.

---

# 27. Recommended completion gates for the Phase 10B agent

At minimum:

1. phase9_identity_verified
2. phase10_selector_verified
3. utility_scaler_verified
4. phase7_identity_verified
5. phase10_artifacts_read_only
6. phase10b_contract_frozen
7. seeds_frozen
8. validation_bank_frozen
9. test_bank_frozen
10. rollout_schedule_frozen
11. optimizer_schedule_frozen
12. population_mix_frozen
13. p10d_both_sides_enforced
14. no_search_training
15. no_phase11_data_used
16. no_phase12_data_used
17. max_30_iterations
18. max_61440_games
19. max_12h_budget
20. phase9_ppo_safety_enforced
21. archive_policy_exact
22. scheduled_validations_complete
23. checkpoint_selection_exact
24. no_post_selection_training
25. final_eval_first_and_only
26. gate_a_recomputed
27. gate_b_recomputed
28. gate_c_recomputed
29. gate_d_recomputed
30. gate_e_recomputed
31. gate_f_recomputed
32. gate_g_recomputed
33. gate_h_recomputed
34. upstream_artifacts_unchanged
35. classification_recomputed
36. full_suite_green

The agent may add stronger implementation gates but may not weaken these.

---

# 28. Handoff to the reviewing chat

When complete, return:

```text
Phase 10B classification
PASS-CANDIDATE / FAIL / BLOCKED

selected checkpoint
path + SHA-256 + model-state digest + iteration

direct P10-D result
EWR + CI vs Phase9+P10-D

neutral rollback result
EWR + CI vs Phase9+neutral

strong-opponent composite
point + CI

belief preservation
CE ratio + top-1 delta

training safety
all counters

budget
games / iterations / wall-clock

upstream preservation
all exact
```

Then stop.

Do not make a production replacement decision.

The reviewing chat decides whether the checkpoint is worth revisiting after the initial Phase 11/12 work.

---

# 29. One-sentence operating rule

> **Adapt the Phase 9 move policy to the already-frozen P10-D setup distribution using the proven Phase 9 PPO/KL machinery, within a strict bounded budget, while preserving Phase 9 and Phase 10 as immutable controls and never delaying the main Phase 11/12 roadmap.**
