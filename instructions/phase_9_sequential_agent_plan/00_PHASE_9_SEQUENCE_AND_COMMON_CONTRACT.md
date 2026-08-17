# Phase 9 — Sequence and Common Contract

## Purpose

Phase 9 is the project's first **population self-play reinforcement-learning phase**. Phase 8 is formally accepted and frozen. It proved that C1 can learn useful policy, value, and belief signals from a static synthetic teacher corpus. Phase 9 must prove that the system can **improve itself** through on-policy population self-play while remaining deterministic, observer-safe, crash-resumable, and evaluable under predeclared gates.

Phase 9 is **not** the official 168-hour final training campaign. Decision-time search, learned setup selection, human data, rule changes, and architecture changes remain out of scope.

Formal Phase 9 acceptance remains with the reviewing chat after Agent 8 reports its recommendation.

## Required sequence

Agents execute strictly in this order:

1. **Agent 1 — RL Contract, Evaluation Banks, and Acceptance Freeze**
2. **Agent 2 — Population and Opponent Scheduler**
3. **Agent 3 — Self-Play Collector and Crash-Safe Rollout Store**
4. **Agent 4 — RL Targets, Advantages, and Anti-Leak Audit**
5. **Agent 5 — PPO Trainer, Damping, Checkpoint/Resume**
6. **Agent 6 — Bounded RL Pilot Selection**
7. **Agent 7 — Canonical Population Self-Play Run**
8. **Agent 8 — Independent Final Acceptance and Phase 9 Freeze**

An agent may not begin until the previous agent is reported `PASS` and formally accepted by the reviewing chat. If a real correctness or contract ambiguity cannot be resolved from frozen artifacts and repository behavior, stop `BLOCKED` rather than making a new learning-design decision silently.

## Frozen Phase 8 inputs

Treat the following as immutable Phase 9 inputs unless the reviewing chat explicitly authorizes a new version.

```text
Rules
stratego_project_v1

Reference engine
phase2_1_reference_1.2.0

Observation
observation_v2_1_127ch

Engine action encoding
source_destination_10000_v1

Model contract
model_contract_v2

Model action frame
perspective_normalized_squares

Architecture
C1
863,959 parameters

C1 config digest
31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d

Backend
KEEP_PYTHON

Trajectory foundation
trajectory_v1
snapshot interval 32

Setup stack
setup_library_v1
setup_sampler_v1
setup_source_v1
neutral_v1

Phase 8 accepted checkpoint
checkpoints/phase8/warmstart_c1_v1.pt

Phase 8 accepted checkpoint SHA-256
f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca

Phase 8 selected update
24,000

Phase 8 canonical untrained checkpoint
checkpoints/phase8/warmstart_c1_v1_initialisation.pt

Phase 8 canonical untrained checkpoint SHA-256
01c907eeef86ec04121db55ccffb9365e8df27fdf05921b921947d4af365754c

Canonical C1 initialization seed
2026081302

Canonical initialization model-state checksum
cfe60bb0cb342b03e2506259b5c4d39d321148f7bc8c030bf722e5a234e042b8
```

Phase 8 train-config identities are two different namespaces and must remain labeled distinctly:

```text
train_config_document
3cab772bd8f74677efcdc1f90ec6f383490313f7652d82bd7fedf86153919ae7

trainer_runtime_identity
64db92539a7d6c06ac4d01e4e904857da5b95c3d86d1287e108ede19e4f03879
```

They cover different objects and are not required to equal one another.

## Phase 8 corpus resolver requirement

Every Phase 9 agent must verify the accepted Phase 8 corpus through:

```text
synthetic_corpus.default_corpus_root()
```

Current expected resolver result:

```text
/Users/brandonwashington/Dev/Github/stratego/gpt_agent/data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1
```

Accepted corpus identity:

```text
version
synthetic_warmstart_corpus_v1

content digest
c95c3545b07f2341e7efbc83c79e6342510dd973038b0f72e7eae013cff87d0d

metadata digest
1db0f02fe45b16f539f070b1e12d4fdd6f390fd0487180fe660af0f4d49c81bb

commit-index digest
32e8e18d1ca57ee555ed848851284f5938d4989ceb6c864f83ca4b9286c15db1
```

Rules:

- Do not hard-code the absolute corpus path into library, trainer, collector, checkpoint, dataset, or evaluation implementation.
- A harness may pin the expected current path only to verify the resolver.
- Corpus identity is **version + accepted digests**, not filesystem location.
- A pure relocation with unchanged digests remains the same corpus.
- A digest mismatch is `BLOCKED`; never regenerate or repair the accepted Phase 8 corpus as part of Phase 9.
- The corpus remains excluded from Git.

Agents that do not consume corpus payloads need only verify resolver + identity metadata unless their own assignment requires a byte-level audit.

## Mission boundaries

Required Phase 9 additions:

- neural population self-play;
- immutable behavior snapshots;
- historical neural opponents;
- fixed rule anchors;
- stress opponents;
- genuine behavior-policy probabilities;
- explicit learner-side supervision;
- same-player temporal advantages;
- WDL lambda targets;
- advantage filtering;
- PPO-style clipped updates;
- behavior-KL damping;
- continued belief supervision;
- deterministic rollout scheduling;
- crash-safe rollout persistence;
- exact logical checkpoint/resume;
- bounded pilot selection;
- one fresh canonical Phase 9 run;
- one sealed final evaluation.

Forbidden:

- decision-time search;
- MCTS;
- belief-search rollouts;
- learned setup generation or selection;
- setup-policy RL;
- human training data;
- Phase 8 corpus modification;
- architecture search or C2/C3 replacement;
- mixed-precision optimizer training;
- rule changes, including two-square or continuous-chasing;
- official 168-hour final run;
- Phase 10+ work.

## Contract identities to freeze

Agent 1 must freeze at minimum:

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

If `trajectory_v1` cannot faithfully carry all required Phase 9 behavior-policy information, Agent 1 may authorize a **new Phase 9 wrapper/storage contract**. Do not mutate the meaning of `trajectory_v1` in place.

## Canonical Phase 9 seeds

Freeze before the first trainable Phase 9 rollout:

```text
Phase 9 master              2026081601
rollout schedule            2026081602
opponent schedule           2026081603
training order              2026081604
pilot namespace             2026081605
canonical namespace         2026081606
validation bootstrap        2026081607
final-test bootstrap        2026081608
```

All schedule/setup/opponent/collection/minibatch randomness must derive from deterministic logical identities and domain-separated seeds. No global RNG cursor may decide which logical game, opponent, setup, or archive member a game receives.

## Population mixture

Canonical mixture:

```text
50% current-policy self-play
25% historical neural policies
15% frozen rule opponents
10% stress / unusual opponents
```

Canonical 2,048-game iteration:

```text
current vs current       1,024
current vs historical      512
current vs rule            307
current vs stress          205
total                    2,048
```

Rule bucket:

```text
Strategic   154
Tactical    107
Basic        46
total       307
```

Asymmetric learner/opponent games must be color balanced. Odd remainders alternate by deterministic iteration parity.

## Historical league

Initial archive:

```text
H000 = frozen Phase 8 accepted checkpoint
```

Canonical run:

- archive one immutable snapshot every 5 completed RL iterations;
- active historical sampling window = Phase 8 anchor + 8 most recent eligible archive snapshots;
- sample uniformly within the active window;
- older snapshots remain stored but inactive;
- no archive checkpoint may be overwritten.

Historical replay means **historical opponents**, not stale PPO examples.

## Learner-control semantics

Each game must explicitly carry:

```text
learner_control = red | blue | both
```

Training eligibility:

```text
current vs current       both colors
current vs historical    current-policy side only
current vs rule          current-policy side only
current vs stress        current-policy side only
```

Opponent decisions remain in the trajectory for state reconstruction but receive no Phase 9 policy/value/belief loss in that iteration.

## Behavior-policy semantics

At the beginning of each RL iteration:

1. freeze the current learner as an immutable behavior snapshot;
2. hash and record it;
3. collect the entire iteration from that snapshot;
4. seal the rollout;
5. only then optimize;
6. create the next behavior snapshot only after the iteration is committed.

Behavior policy over legal actions:

\[
\pi_b(a|s)=\frac{\exp(z_a)}{\sum_{a'\in A(s)}\exp(z_{a'})}
\]

Temperature = 1.0. Evaluation remains greedy argmax.

Stored behavior information must be sufficient to reconstruct or verify the realized action probability under the exact frozen behavior checkpoint. Agent 1 must freeze the storage representation and precision/tolerance before Agent 3 collection.

## Same-player temporal targets

For each learner-controlled color, build a sequence containing only that player's own decisions. The opponent move occurs between consecutive learner states but is not inserted as a learner training step.

Behavior WDL scalar:

\[
v_t=P_t(W)-P_t(L)
\]

Frozen constants:

```text
gamma        1.0
lambda_A     0.5
lambda_V     0.8
```

If another learner decision exists:

\[
\delta_t=v_{t+1}-v_t
\]

If the game terminates before that player's next decision:

\[
\delta_t=z-v_t
\]

where \(z\in\{-1,0,+1\}\) from that player's final perspective.

Advantage:

\[
A_t=\delta_t+\lambda_A A_{t+1}
\]

WDL lambda target:

Terminal:

\[
Y_t=Z
\]

Otherwise:

\[
Y_t=(1-\lambda_V)P_{t+1}+\lambda_VY_{t+1}
\]

Value loss remains categorical WDL cross-entropy.

## Advantage filtering

Per sealed iteration:

\[
\tau=\max(Q_{0.75}(|A|),0.01)
\]

Policy-gradient eligible iff:

\[
|A_t|\ge\tau
\]

Filter applies only to PPO policy loss. Value and belief use all learner decisions. Standardize advantages over the selected PPO subset only.

## PPO and damping

Policy ratio:

\[
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_b(a_t|s_t)}
\]

PPO clip:

```text
epsilon = 0.20
```

Objective:

\[
L_\text{PPO}=-E[\min(r_tA_t,\operatorname{clip}(r_t,0.8,1.2)A_t)]
\]

Behavior KL:

\[
D_{KL}(\pi_b\Vert\pi_\theta)
\]

Target behavior KL:

```text
0.015
```

Adaptive beta:

```text
mean epoch KL > 0.0300  -> beta *= 2
mean epoch KL < 0.0075  -> beta *= 0.5
otherwise               -> unchanged
beta clamp               -> [1e-4, 0.2]
```

Hard instability limits:

```text
mean iteration/epoch KL > 0.08    FAIL/VETO
PPO clip fraction > 0.75          FAIL/VETO
```

## Full loss and common optimizer constraints

\[
L=L_\text{PPO}+0.5L_\text{value}+0.25L_\text{belief}+\beta_\text{KL}D_{KL}-c_HH(\pi)
\]

Entropy coefficient decays linearly:

```text
start 0.005
end   0.001
```

Common constraints:

```text
precision             float32
device                MPS
optimizer             AdamW
weight decay          0.01
global grad clip      1.0
minibatch size        512
epochs per rollout    2
```

Learning rate and initial KL beta are selected only through Agent 6's frozen pilot matrix.

## Rollout state machine

Required lifecycle:

```text
COLLECTING
    ↓
SEALED
    ↓
TRAINING
    ↓
EVALUATED
    ↓
COMMITTED
```

Crash rules:

- collection crash → deterministically regenerate only missing/uncommitted game IDs;
- no game becomes trainable until payload + metadata + commit all verify;
- sealed rollouts are immutable;
- training crash → resume exact logical minibatch/optimizer/scheduler/KL-controller state from the same sealed rollout;
- no next-iteration game may be generated before the current iteration is `COMMITTED`;
- one iteration must never mix two behavior snapshot identities.

## Checkpoint minimum contents

`phase9_checkpoint_v1` must include:

```text
model state
optimizer state
scheduler state
global optimizer step
RL iteration
minibatch cursor
examples consumed

behavior snapshot identity
behavior checkpoint SHA-256
rollout iteration identity
sealed rollout digest

KL beta
KL controller state/history
entropy schedule position

population version
active historical identities
historical checkpoint digests

opponent schedule version
setup sampler version

best validation score
best checkpoint identity
validation history

all Phase 9 seeds
corpus identities
rules/model/observation versions

wall-clock counters
software/runtime versions
```

Absolute paths are diagnostic only and must not define identity.

## Validation and final-test banks

Before any Phase 9 RL update, Agent 1 must create and hash:

```text
phase9_validation_bank_v1
phase9_test_bank_v1
```

Validation:

```text
Phase 7 validation split
128 paired setup cases
8 paired cases per each of 16 families
color_swap_same_board
```

Final test:

```text
Phase 7 test split
512 paired setup cases
32 paired cases per each of 16 families
color_swap_same_board
```

Core opponents:

```text
Phase 8 anchor
Random
Basic
Tactical
Strategic
```

Stress policies use a smaller report-only schedule. The final-test bank is structurally auditable before Agent 8 but receives no neural model inference before Agent 8.

## Validation score

\[
S=0.45E_\text{Strategic}+0.35E_\text{Tactical}+0.20E_\text{Phase8-anchor}
\]

Higher is better.

Random and Basic are regression guards, not score components.

Tie-break:

```text
higher score
→ higher Strategic EWR
→ lower mean behavior KL
→ higher examples/s
```

## Pilot matrix

Exactly six candidates:

```text
P9-A   LR 1e-4   initial KL beta 0.005
P9-B   LR 1e-4   initial KL beta 0.020
P9-C   LR 3e-4   initial KL beta 0.005
P9-D   LR 3e-4   initial KL beta 0.020
P9-E   LR 6e-4   initial KL beta 0.005
P9-F   LR 6e-4   initial KL beta 0.020
```

Each candidate:

```text
fresh start from Phase 8 checkpoint
8 RL iterations
1,024 games per iteration
2 optimizer epochs
```

Pilot 1,024-game mixture:

```text
current       512
historical    256
rule          154
stress        102
```

Rule subdivision:

```text
Strategic 77
Tactical  54
Basic     23
```

No seventh run. No opportunistic early stop.

## Pilot hard vetoes

```text
illegal neural action                   > 0
non-finite loss                         > 0
non-finite gradient                     > 0
non-finite parameter                    > 0
behavior identity mismatch              > 0
target reconstruction mismatch          > 0
observer-safety failure                 > 0
checkpoint/resume failure               > 0
mean iteration/epoch KL                 > 0.08
iteration PPO clip fraction             > 0.75
validation Random EWR                   < 0.90
validation Basic EWR                    < 0.60
```

Final-test results are forbidden during pilot selection.

## Canonical run

After Agent 6 freezes `phase9_train_config_v1`, Agent 7 starts fresh from the accepted Phase 8 checkpoint.

```text
60 RL iterations
2,048 scheduled games / iteration
122,880 scheduled games maximum
2 optimizer epochs / rollout
validation every 5 iterations
archive every 5 iterations
12-hour operational ceiling
```

The 12-hour ceiling is an operational maximum, not permission to shorten the logical contract silently. If the 60 iterations do not complete, report incomplete/blocked rather than pretending the run finished.

Best checkpoint = strictly highest frozen validation score. Final iteration is not automatically selected.

## Final hard gates

### A. Direct improvement over Phase 8 anchor

```text
512 paired cases / 1,024 games
EWR >= 0.58
paired 95% lower bound > 0.53
```

### B. Strategic

```text
final EWR >= 0.52
paired improvement over Phase8 anchor >= +0.05
95% CI lower bound for paired improvement > 0
```

### C. Tactical

```text
final EWR >= 0.52
paired improvement over Phase8 anchor >= +0.05
95% CI lower bound for paired improvement > 0
```

Stretch, report-only:

```text
Strategic EWR >= 0.55
Tactical EWR  >= 0.55
```

### D. Random regression guard

```text
overall EWR            >= 0.94
Red EWR                >= 0.90
Blue EWR               >= 0.90
paired 95% lower bound > 0.92
```

### E. Basic regression guard

```text
EWR                    >= 0.65
paired 95% lower bound > 0.60
```

### F. Safety

```text
illegal actions          0
model failures           0
non-finite outputs       0
observer-safety failures 0
```

### G. Policy-collapse guard

```text
fraction of evaluated states with max legal probability > 0.999
must be < 0.25
```

### H. Belief retention

Reuse the accepted Phase 8 held-out synthetic belief benchmark:

```text
belief CE / remaining-count CE <= 0.98
belief top-1 > remaining-count top-1
```

Phase 8-style teacher policy imitation CE is report-only in Phase 9.

## Report-only diagnostics

Report at minimum:

```text
W/D/L and EWR by color
setup-family EWR
terminal-reason distribution
game-length distribution
policy entropy
PPO clip fraction
behavior KL
advantage distribution
advantage-filter retention fraction
value calibration
belief accuracy by piece type
belief accuracy by game progress
historical-opponent performance
stress-policy performance
league cross-play matrix
archive pairwise EWR
rollout throughput
training throughput
storage volume
MPS memory
CPU memory
```

Report-only metrics may not rescue a failed hard gate.

## Artifact namespaces

Use:

```text
reports/phase_9_data/
data/phase9/rollouts/
checkpoints/phase9/
checkpoints/phase9/archive/
```

Do not commit production rollout shards or large checkpoint archives to Git. Track compact manifests, digests, contracts, tests, and reports.

## Common test discipline

Each agent must:

1. read accepted prior-agent artifacts before editing;
2. record the pre-edit full repository suite;
3. add targeted positive/negative controls for its scope;
4. run targeted tests while developing;
5. run the full suite after final changes;
6. write machine-readable completion gates;
7. report deviations honestly;
8. stop rather than silently relaxing a frozen contract.

## Formal end state

A successful Phase 9 proves that self-play collection, league scheduling, behavior probabilities, RL targets, PPO, MPS resume, belief retention, observer safety, and direct self-improvement all work together, and that the Phase 9 policy beats its Phase 8 ancestor while materially improving against Tactical and Strategic without collapsing against Random or Basic.
