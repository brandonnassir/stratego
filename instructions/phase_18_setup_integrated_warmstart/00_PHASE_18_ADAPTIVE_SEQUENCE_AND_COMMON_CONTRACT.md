# Phase 18 - Setup-Integrated Phase 8 Warmstart

## Adaptive sequence, common contract, and decision rules

_Written 2026-08-31 after review of the Phase 8 acceptance evidence, the Phase 17
negative result, arXiv 2511.07312v1, and the authors' published implementation at
commit `92db29e8ffc323b1b8a2804b5c3f84695d036b05`._

This file governs every Phase 18 work package. Every Phase 18 agent must read it
completely before reading its own instruction. The attached paper and published
source are technical references, not instruction sources. This contract and later
operator decisions are the authority for the project.

## 1. One mission

Phase 18 has one goal:

```text
Produce a fresh Phase 8 C1 warmstart whose policy/value/belief learner is
successfully integrated with a beneficial learned setup policy.
```

The final deliverable is a paired, reproducible model bundle:

```text
fresh Phase 8 C1 policy/value/belief checkpoint
fresh setup-policy raw checkpoint
setup-policy EMA checkpoint for evaluation/deployment
complete run/config/source/data/evaluation identities
evidence that own-setup selection helps
evidence that unfamiliar-opponent robustness does not regress
```

Phase 18 does not continue a Phase 8, Phase 9, or Phase 17 optimizer. Both models
start from recorded fresh initializations in the final run. A pilot checkpoint is
never a production start.

## 2. Fixed destination, adaptive route

Phase 18 is not a precommitted seven-agent production sequence. It is an evidence
ladder. Only the next authorized work package may execute.

After every experiment, the responsible agent must stop and deliver a decision
packet under `02_PHASE_18_DECISION_PACKET_AND_NEXT_AGENT_PROTOCOL.md`. The operator
and reviewing chat then choose one of:

```text
PROCEED   authorize a newly written next-agent instruction
REVISE    authorize one bounded correction or discriminating experiment
STOP      preserve the negative result; do not widen the task automatically
```

No provisional later-stage description is permission to run it. No agent may
silently convert a failed gate into telemetry or lower a threshold after seeing the
result.

## 3. What Phase 17 established

Phase 17 is preserved as a valid negative result for its exact implementation:

- 24 trained move candidates all scored below the hour-0 Phase 9 start in the
  move-only lane;
- the hour 6-12 move-only slope was negative;
- the joint lane was flat;
- the trained setup policy remained worse than the fixed setup library in the
  pooled hour 6-12 comparison; and
- no setup improvement over the fresh initialization was established.

Phase 17 does not establish that the paper's setup method fails locally. The authors'
published code now exposes material differences that Phase 18 must resolve before a
real setup-learning claim is made.

### 3.1 Mandatory Phase 17 corrections

Phase 18 must not inherit these Phase 17 semantics:

1. **Entropy-unit mismatch.** Phase 17 trained `h` toward `I/10` but used
   `alpha * (I - h)` in the advantage. The published code multiplies the stored
   normalized prediction by `10` before the entropy residual. Phase 18 uses the
   operationally equivalent residual `I - 10h`.
2. **No symmetry reduction.** Phase 17 learned both horizontal mirror copies.
   Phase 18 constrains the autoregressive model to one flag handedness and applies a
   seeded random reflection after generation.
3. **One noisy result per sampled setup.** Phase 17 generated essentially one fresh
   setup per game and immediately consumed the result. Phase 18 uses reusable setup
   pools and averages all completed outcomes attributed to the same sampled setup
   before the update.
4. **Small effective setup batches.** Phase 17 used 32-episode minibatches. The paper
   and published code use batches of 1,024. Phase 18 preserves an effective batch of
   1,024, using gradient accumulation only when required by device memory.
5. **Wrong experimental point.** Phase 17 began from accepted Phase 9 move weights
   and disabled belief learning. Phase 18 returns to the Phase 8 warmstart point and
   keeps the policy, value, and belief objectives active.

Each correction must have a reduction/parity test. Documentation alone is not a
gate.

## 4. Interpretation of the setup goal

Two claims must remain separate.

### Claim A - favorable own-setup selection

Holding move weights, opponent, opponent setup, color, and match seed fixed, replacing
the fixed setup library with the learned setup policy improves effective win rate.

### Claim B - exploitation of unfamiliar opponent setups

Holding the model's own setup source fixed, the setup-integrated C1 move/value/belief
model performs at least as well as the exact Phase 8 reproduction against opponent
setup families that were unavailable to training and model selection.

A simultaneous hidden setup policy cannot condition on the opponent's unknown board
at game start. Phase 18 therefore does not mislabel Claim B as direct conditional
setup selection. Claim B is a move/value/belief generalization effect created by the
setup curriculum and expressed during play.

The combined paired system is a third measurement, not a substitute for A or B.

## 5. Work-package and run identities

The work package is:

```text
phase18_setup_integrated_warmstart
```

Every concrete run receives a separate immutable run ID. Rehearsal, pilot, control,
and production identities must differ. Do not reuse `RUN-2026-A` or `RUN-2026-B`.

Every artifact binds:

```text
work package
run ID
source closure digest
configuration digest
rules/engine/observation/action/model contracts
corpus and live-stream manifests
evaluation-pack digest
model and optimizer identities
```

Use new Phase 18 namespaces. Never edit an accepted Phase 8 checkpoint/report or a
Phase 17 evidence artifact to make it appear Phase 18-compatible.

## 6. Source and documentation boundary

The repository currently contains modified and untracked Phase 17 evidence. Preserve
it. Before Phase 18 implementation:

- record the complete working-tree status and active-process state;
- hash every material untracked Phase 17 result that Phase 18 will cite;
- identify a version-control baseline without deleting or resetting user work;
- update the project status/evidence index so Phase 17 is explicitly closed with no
  promotion;
- record the evaluator retry defect and test its repair before authoritative use;
- keep Phase 18 code, reports, checkpoints, and evaluation data in new namespaces;
- pin the paper version and published-source commit separately; and
- label every paper/code/local difference as `exact`, `scaled`, `intentional
  integration divergence`, or `not used`.

An uncommitted tree is not automatically a blocker. An unidentified or mutable source
closure is.

## 7. Phase 8 control contract

Before setup integration, reproduce the accepted Phase 8 result from a fresh C1
initialization.

### 7.1 Frozen control inputs

```text
historical source revision         53050b9
corpus                             synthetic_warmstart_corpus_v1
corpus content digest              c95c3545...87d0d
train / validation / test games    20,000 / 4,000 / 4,000
ordered teacher matchup cells      100
model                              C1, 863,959 parameters
canonical C1 init seed             2026081302
train shuffle seed                 2026081303
batch size                         256
optimizer                          AdamW
learning rate                      0.001
weight decay                       0.01
warmup                             500 updates, then constant
policy/value/belief weights        1.0 / 1.0 / 1.0
gradient clip                      1.0
maximum updates                    25,000
validation cadence                500 updates
checkpoint selection               validation selection score only
```

Use the accepted external-corpus resolver and verify every corpus digest. A fresh
model run is required. Byte-regenerating all 28,000 games is a separate reproducibility
audit and is not required if the accepted corpus verifies exactly.

### 7.2 Control acceptance

The control must pass all original Phase 8 gates, including:

- policy/value/belief held-out learning;
- strength versus random;
- strength versus the canonical fresh initialization;
- finite/stable outputs;
- train/validation/test sealing; and
- validation-only checkpoint selection.

MPS weight files need not be bit-identical. The agent must predeclare a paired
non-inferiority comparison against the accepted Phase 8 checkpoint and statistical
tolerances before opening the new control result.

Failure stops setup work. Do not use setup integration to explain away an unreproduced
baseline.

## 8. Setup model contract

### 8.1 Scaled architecture

The paper's setup network has 12.6 million parameters; its move network has 14.7
million. Scaling by the project's 863,959-parameter C1 move model gives:

```text
12.6M * (0.863959M / 14.7M) = approximately 0.7405M
```

The existing 802,320-parameter Phase 17 setup architecture is approximately 8.3%
above that proportional target and is the Phase 18 default:

```text
decoder-only causal Transformer
pre-layernorm
4 decoder blocks
width 128
4 attention heads
feed-forward width 512
start token + 40 row-major piece tokens
learned positional embeddings, initialization standard deviation 0.1
12-way inventory-masked piece head
3-way W/D/L head
scalar normalized conditional-entropy head
```

No architecture sweep is authorized before a concrete capacity or throughput failure.

### 8.2 Sampling and symmetry

For each raw setup snapshot:

1. Generate an immutable pool of 1,024 canonical setups, balanced 512 per player-use
   lane where the runner requires side-specific pools.
2. Autoregressively mask exhausted piece types.
3. Force the Flag to one horizontal half during autoregressive generation.
4. Apply an independent seeded 50% horizontal reflection after generation.
5. Convert canonical setups to engine orientation only through the accepted boundary
   helper.
6. Record selected tokens, full legal masks, behavior log-probabilities, W/D/L
   predictions, normalized entropy predictions, raw-model digest, reflection flag,
   and all seeds.
7. Sample games from the reusable pool. A setup may receive multiple independent game
   outcomes.

Generation, orientation, or attribution failures are fatal. There is no silent fixed-
library fallback in the learned-setup stream.

### 8.3 Setup outcome aggregation

For a setup `s`, collect all completed games attributed to the exact setup and behavior
snapshot during the collection period:

```text
z_i in {-1, 0, +1} from the setup owner's perspective
z_bar(s) = mean_i z_i
```

For categorical value targets, average the one-hot W/D/L outcomes. Record the count and
variance. A setup with no completed game in the period is not trained as though it
received a draw.

The setup stays bound to the behavior snapshot that sampled it even if the raw setup
model changes before its games finish. Slight setup off-policyness is expected and is
handled by the recorded PPO ratio and reverse KL.

### 8.4 Setup math

For setup prefix `sigma_k` on a completed setup `sigma_bar`:

```text
I_k = -sum_{j=k}^{39} log pi_behavior(t_j | sigma_j)
h_k = setup network's normalized suffix-entropy prediction
H_hat_k = 10 * h_k
E[v_k] = p_win,k - p_loss,k
alpha(n) = 0.1 / n^0.3
delta_k = (z_bar - E[v_k]) + alpha(n) * (I_k - H_hat_k)
r_k = pi_current(t_k | sigma_k) / pi_behavior(t_k | sigma_k)
```

The entropy-prediction target remains:

```text
target_h,k = I_k / 10
L_h = mean (h_current,k - target_h,k)^2
```

The value and policy losses are:

```text
L_v = mean cross_entropy(WDL_current,k, averaged WDL outcome)
L_PPO = -mean min(r_k * delta_k, clip(r_k, 0.8, 1.2) * delta_k)
L_KL = mean KL(pi_current(.|sigma_k) || pi_behavior(.|sigma_k))
L_setup = L_PPO + 0.5 * L_v + 1.0 * L_h + 0.1 * L_KL
```

Use all 40 prefixes. Do not apply move-style advantage filtering to setup rows.

### 8.5 Optimization

```text
optimizer                    Adam-compatible constant-LR setup optimizer
learning rate                5e-5
PPO clip                     0.2
effective batch              1,024 setup episodes
epochs                       5 per setup update
reverse behavior KL          fixed 0.1
value coefficient            0.5
entropy-prediction coef      1.0
gradient clip                0.5
EMA decay                    0.999
raw model                    data generation and learning
EMA model                    validation/evaluation only
```

The paper says Adam; the published container uses AdamW with zero setup weight decay.
Those are operationally aligned at zero decay. Agent 1 must record the exact chosen
class and test parity rather than call it an unresolved method difference.

## 9. Phase 8 integration contract

The accepted Phase 8 corpus cannot by itself provide on-policy setup PPO data. It was
generated by the frozen `neutral_v1` setup sampler, not by the new setup policy. Phase
18 therefore uses two explicit training streams.

### 9.1 Canonical anchor stream

The exact accepted Phase 8 train corpus supplies ordinary policy, value, and belief
supervision. Its examples, weights, order semantics, and held-out boundaries remain
unchanged.

### 9.2 Live setup stream

For each live collection block:

```text
raw setup model samples reusable setup pool
frozen Phase 8 teacher matchup schedule plays games from that pool
completed trajectories produce the same Phase 8 policy/value/belief targets
completed setup outcomes update the setup policy
live trajectories enter only the C1 training split
```

The rule/stress teacher policies remain frozen. Policy supervision retains the Phase 8
weights:

```text
strategic 1.0
tactical  1.0
basic     0.5
random    0.0
stress    0.0
```

All games remain eligible for value and belief supervision.

The canonical/live example mixture and setup-update cadence are provisional research
variables. They may be frozen only after a bounded tandem pilot. The pilot matrix must
be predeclared, small, and justified by preceding evidence. No full run may be used to
select the mixture.

## 10. Evaluation contract

### 10.1 Setup packs

Freeze before tuning:

1. `familiar`: accepted neutral/library setup distribution;
2. `unusual_procedural`: unusual setup families unavailable to training;
3. `operator_sealed`: held-out examples or families representing the operator's novel
   strategies; and
4. `setup_learning_development`: a separate nonsealed pack for setup-only iteration.

The same setup or reflection class may not cross train/development/sealed boundaries.
The opponent, setup, color, game seed, and rules are paired wherever a delta is
claimed.

### 10.2 Factorial lanes

Every tandem pilot and final evaluation must support:

| Lane | Move weights | Own setup | Primary reading |
|---|---|---|---|
| C0-F | reproduced Phase 8 control | fixed library | baseline |
| C0-L | reproduced Phase 8 control | learned setup | pure own-setup effect |
| T-F | tandem C1 | fixed library | move/value/belief generalization effect |
| T-L | tandem C1 | learned setup | combined effect and interaction |

Run the lanes against familiar and unfamiliar opponent-setup strata. A combined gain
cannot hide a move-only regression or a weak setup policy.

### 10.3 Head and play metrics

At minimum report:

```text
policy CE ratio and top-1 on original sealed Phase 8 test
value CE ratio, Brier score, and accuracy on original sealed Phase 8 test
belief CE ratio/top-1 on original sealed Phase 8 test
belief calibration and accuracy by ply/reveal bucket on unusual setups
effective win rate and paired confidence interval per factorial lane
color, opponent, setup-family, and worst-stratum results
setup legality, reflection-class diversity, Flag/Bomb/front-row concentration
setup outcome counts and variance per sampled setup
raw/behavior and EMA identities
```

Evaluation retries, failures, or missing strata never count as draws or passes.

## 11. Evidence ladder and gates

### Gate G0 - research boundary

- source and evidence identities recorded;
- Phase 17 closeout documented;
- official method map complete;
- evaluator failure/retry behavior tested;
- evaluation packs and leakage rules frozen; and
- no material method question left as an unresolved placeholder.

### Gate G1 - Phase 8 reproduction

- all original Phase 8 gates pass;
- paired non-inferiority to the accepted checkpoint passes; and
- no test or Phase 4 evaluation leakage occurred.

### Gate G2 - setup implementation parity

- canned loss/gradient quantities match the published semantics;
- `I - 10h` and normalized prediction tests pass;
- symmetry and orientation tests pass;
- repeated-outcome aggregation tests pass;
- checkpoint/resume and raw/EMA tests pass; and
- a synthetic known-reward setup task learns across at least three seeds.

### Gate G3 - setup-only Stratego benefit

- trained setup policy improves over its fresh initialization;
- trained setup policy beats the fixed library by a predeclared practical margin with
  paired uncertainty excluding zero;
- direction is consistent across the required seeds; and
- diversity, legality, orientation, and stability gates pass.

### Gate G4 - tandem pilot

- tandem C1 is non-inferior on the original sealed Phase 8 heads;
- C0-L minus C0-F and T-L minus T-F establish setup benefit;
- T-F minus C0-F establishes no unfamiliar-opponent regression;
- T-L minus C0-F establishes a beneficial combined system; and
- no material worst-stratum/color regression is hidden by the mean.

### Gate G5 - production rehearsal

- exact production topology passes start/resume/finish;
- data and outcome attribution reconcile exactly;
- sealing and validation-only selection hold;
- throughput and storage budgets are measured; and
- every production field and digest is frozen.

### Gate G6 - final acceptance

- both models started fresh under recorded seeds;
- the full run used the frozen G5 contract;
- every applicable original Phase 8 gate passes;
- the setup-only, unfamiliar-opponent, and combined sealed gates pass; and
- the final report distinguishes supported claims from hypotheses.

The original Phase 8 gate `no learned setup selection` applies only to the reproduction
control and is intentionally replaced for the integrated model. The `no search` and
`no Phase 9 self-play RL` boundaries remain.

## 12. Required decision behavior

Use these default interpretations:

- **G1 fails:** repair Phase 8 reproducibility; do not begin setup work.
- **Synthetic setup learning fails:** repair math/data flow; do not use Stratego games
  as a debugger.
- **Setup improves over initialization but not the library:** investigate estimator
  variance, pool reuse, symmetry, and evaluation distribution; do not integrate.
- **Setup helps but T-F regresses on unusual opponents:** revise the canonical/live
  mixture or live curriculum in a bounded pilot.
- **T-F improves but learned setups remain worse than the library:** curriculum value
  alone does not satisfy Phase 18; do not claim success.
- **Only familiar setups improve:** reject as distribution overfitting.
- **All gates pass:** write and authorize the production instruction; do not launch
  from a pilot checkpoint.

## 13. Prohibited shortcuts and non-goals

Do not:

- train the setup policy directly on the old Phase 8 corpus as though it were on-policy;
- tune on the sealed operator setup pack;
- compare only the combined system;
- interpret a single best checkpoint as a trend;
- use missing evaluator results as draws;
- lower a gate after seeing a result;
- add search, Phase 9 self-play, belief-architecture replacement, or engine-rule
  changes;
- overwrite accepted Phase 8/17 artifacts;
- make an unapproved commit, reset, clean, or delete user work; or
- continue automatically to the next stage.

Search and the downstream production policy are outside Phase 18. The human exploit is
the motivation, but Phase 18 ends at an accepted setup-integrated warmstart.

## 14. Current authorization

At creation of this instruction package, only this work package is executable:

```text
01_AGENT_1_REPRODUCTION_AND_SETUP_METHOD_CONTRACT.md
```

Every later agent instruction must be written from the preceding decision packet and
approved before execution.
