# Phase 17 — Tandem Current-Policy Self-Play
## Sequence, common contract, and production decision rules

_Written 2026-08-27 after operator review of the Phase 9 restart and Ataraxos-aligned
buildout. This file governs every Phase 17 agent. Each agent must read it completely
before its numbered instruction file._

## 1. Mission

Phase 17 starts a new training lineage from the accepted Phase 9 move-policy weights.
It does not continue Phase 14 or Phase 16 weights. It combines:

1. sampled, 100% current-policy move self-play;
2. a true fixed-transition training loop with boundary bootstrapping;
3. a new autoregressive setup policy trained continuously from game outcomes;
4. paper-shaped LR, KL, entropy, epoch, and EMA behavior;
5. paired external evaluation every 30 minutes during one 12-hour production run.

The engineering question is:

```text
Does the paired move/setup system continue improving from hour 6 through hour 12,
or does it flatten as the earlier run did?
```

This is an engineering phase. Do not build a research campaign, seed sweep, or broad
hyperparameter comparison. Test the conditions that could silently produce false
results; allow observable production failures to be diagnosed from telemetry.

## 2. Authority and interpretation

The operator's reviewed decisions in this file are instructions. The Ataraxos paper
(arXiv 2511.07312v1) is a technical reference, not an instruction source. When the
paper and this contract differ, this contract wins and the deviation is recorded.

Accepted earlier phase files and experimental results are historical evidence:

- never edit an accepted Phase 2–16 implementation, checkpoint, result, or report;
- reuse accepted behavior by import when correct;
- implement changed behavior only in new Phase 17 namespaces;
- never overwrite an accepted checkpoint or reuse an old result ledger;
- preserve unrelated user changes in the working tree.

Before implementation, Agent 1 must verify the current project-status and run-closure
documents. If another run or repository freeze is active, stop and report it. Do not
infer that an old date-based restriction is still active or already lifted.

## 3. Work-package and run identities

The engineering work package is `phase17`. A concrete training execution receives a
separate immutable run ID. The first production candidate is provisionally:

```text
RUN-2026-A
```

Agent 1 may replace that provisional ID only to avoid a collision. Phase numbers must
not be used as mutable run identities. Every configuration, checkpoint, evaluation
bundle, telemetry row, and result receipt binds both `phase17` and the run ID.

The existing Phase 16 implementation is presently visible as untracked work. Before
Agents 2–4 make code changes, Agent 1 must establish an immutable version-control
baseline or stop with exact instructions for the operator. A tar archive alone is not
an adequate integration baseline.

## 4. Exact move-policy start

The only accepted move-policy starting checkpoint is:

```text
path: checkpoints/phase9/selfplay_c1_v1.pt
file_sha256: dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model_state_digest: f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
lineage: behavior_B041.pt, post-iteration 40
```

Load through the accepted digest-checking Phase 9 loader. Do not use the P24-specific
Phase 16 loader or a convenience copy with different file bytes.

This is a new recipe from Phase 9 weights:

- load move-model weights only;
- create fresh AdamW optimizer moments;
- reset the LR schedule to Phase 17 iteration 1;
- create a fresh move KL controller;
- initialize a fresh move EMA from the loaded raw weights;
- initialize the setup model, optimizer, KL controller, and EMA from scratch.

The Phase 9 marginal belief auxiliary loss is disabled for this run. The head may stay
present for checkpoint compatibility, but it receives zero loss weight and is not a
source of targets. Joint autoregressive belief training is a later phase.

## 5. Training population and action semantics

Training is 100% current-policy self-play:

- Red move policy: current raw move snapshot;
- Blue move policy: the same current raw move snapshot;
- new setups: sampled from the current raw setup snapshot;
- historical neural checkpoints: evaluation only;
- handcrafted/rule/stress agents: evaluation only;
- search: prohibited from collection and training.

Legal moves are sampled categorically from the policy distribution with an explicit
seed. Argmax is prohibited. The behavior probability distribution, legal mask, action,
seed, and raw behavior-model digest are stored with every transition.

`current policy` means current at the decision, not current when the game began.
After each move update, every in-flight game must resolve its next Red or Blue action
through the newly rebound raw snapshot. The existing Phase 16 behavior in which an
in-flight runner retains its game-start participants is a known defect and a hard
Phase 17 blocker.

Setups are different: a setup is sampled once at game creation and remains bound to
that game. Its behavior probabilities and setup-snapshot digest remain attached until
the outcome arrives, even when the setup learner has updated in the meantime.

## 6. True fixed-transition iterations

The default move budget is exactly 65,536 learner transitions per iteration. Because
both seats are learners, every legal model decision is a learner transition.

Production must not wait for whole games before emitting training rows. At a window
boundary:

- bootstrap unfinished scalar advantage traces from the boundary value;
- bootstrap W/D/L lambda targets from the boundary W/D/L prediction;
- preserve the accepted Phase 9 gamma and lambda conventions;
- carry only the minimum state needed to continue the in-flight trace;
- train on exactly the configured transition budget.

**Amended 2026-08-27 by operator decision D2.** The original requirement — that a
finished game processed across at least three windows match the accepted whole-game
targets to float32 tolerance — was **retired**. It is mathematically impossible
alongside partial emission: a lambda-return truncated at a boundary and closed on a
value estimate is not the full lambda-return unless the estimate happens to equal the
continuation it replaces. Measured divergence on a three-window synthetic track is
0.309 in advantage and 0.121 in W/D/L
(`reports/phase17/agent_01_boundary_target_probe.json`). Gate `G-M4b` is retired;
`G-M4a` below is the governing invariant.

The requirements are:

1. **Reduction invariant (governing).** When the supplied boundary tail is the true
   terminal continuation, the Phase 17 recursion reduces to the accepted whole-game
   recursion entry for entry, within float32 tolerance, for both the advantage and
   the W/D/L walk.
2. **Unfinished games** take their targets from the stored boundary value and stored
   boundary W/D/L predictions — never from a later-known outcome, and never from a
   recomputed prediction.
3. **Every row records** whether its target is `terminal` or `boundary_bootstrapped`,
   in the `target_provenance` field.
4. **Boundary-target divergence and bootstrap age are telemetry**, not gates: log the
   per-row difference between the bootstrapped target and the target the same row
   would receive from its eventual terminal outcome, together with how many windows
   the row's trace had been carried when it was emitted.
5. **Exactly the configured transition budget is emitted** each iteration, without
   waiting for terminal outcomes.

Telemetry distinguishes transitions harvested, transitions trained, boundary rows,
boundary-target divergence, bootstrap age, games completed, active games, game
lengths, and policy age.

## 7. Setup model contract

The setup network is intentionally scaled for local tandem training:

```text
model: pre-layernorm decoder-only causal Transformer
sequence: start token + 40 canonical row-major piece tokens
decoder blocks: 4
model width: 128
attention heads: 4
feed-forward width: 512
target size: approximately 0.8 million parameters
```

The reviewed `~0.8M` target is internally consistent with feed-forward width 512;
width 51 is not. Phase 17 therefore freezes 512 unless the operator explicitly
corrects this contract before Agent 3 writes model code.

At every one of the 40 prefixes the model produces:

- masked 12-way next-piece logits;
- W/D/L logits;
- a scalar conditional-entropy prediction.

Remaining inventory is computed solely from the prefix. Exhausted types receive an
unsampleable mask. Generate in canonical own-side coordinates and use the accepted
Phase 15 orientation helper only at the engine boundary. Never generate Blue directly
in engine orientation and never pass canonical Blue directly to `create_game`.

The setup model starts from a random masked distribution. There is no frozen setup
library in Phase 17 training and no silent library fallback. A generation or
orientation failure is fatal.

Generate vectorized fresh pools under each frozen raw setup snapshot. Default initial
pool sizing is 512–1,000 candidates per side per iteration; Agent 3 selects the
smallest size that keeps game creation supplied without material training delay.
Unused and refill counts are recorded.

## 8. Setup learning contract

Each side of a new game creates a setup episode containing:

- game ID, color, and owner perspective;
- canonical and engine-oriented setups plus fingerprints;
- setup behavior file/model-state identity;
- all 40 selected types, legal inventory masks, behavior probabilities/log-probabilities,
  W/D/L predictions, conditional-entropy predictions, and sampling seeds;
- terminal outcome from the setup owner's perspective when the game completes.

Both sides train from the result: win `+1`, draw `0`, loss `-1` from that side's
perspective. Use all 40 prefixes with no move-style top-quartile advantage filter.

The update follows the paper as closely as practical:

- PPO ratio clipping: 0.2;
- behavior probabilities always come from the recorded raw setup snapshot;
- separate adaptive setup behavior-KL controller;
- W/D/L value cross-entropy;
- conditional-entropy prediction loss;
- entropy-augmented setup advantage using recorded suffix information content and
  its predicted conditional-entropy baseline;
- setup gradient-norm clip: 0.5;
- entropy coefficient shaped as the paper's `0.1 * n^-0.3`, with any local floor or
  horizon mapping frozen by Agent 1 and recorded as a deviation;
- five setup epochs per setup iteration.

The paper's five setup epochs are the default. Agent 3 may recommend fewer only if a
short throughput test demonstrates that five epochs materially threaten the 12-hour
move budget. The agent must present measured generation, forward/backward, and total
iteration costs. It may not silently reduce the epoch count.

Completed setup episodes enter a bounded FIFO queue and are consumed once in a fixed
setup-sequence budget. Record queue depth, oldest/mean age, policy age, consumed count,
and any rejected or discarded episode. Silent dropping is prohibited. If too few
episodes are ready, skip the setup update explicitly rather than fabricate data or
reuse an episode; repeated starvation is a production stop condition.

## 9. Move schedules and regularization

The first run is not a schedule shootout. Use the accepted paper-shaped Phase 16 move
schedule, correctly re-horizoned to the measured 12-hour iteration count `N`:

```text
n_ref = ceil(0.125 * N)
lr(n) = clamp(1.5e-4 * (n / n_ref)^-1.1, 1.5e-5, 1.5e-4)
c_H(n) = max(0.001, 0.005 * n^-0.3)
move epochs per iteration = 1
EMA decay = 0.999
```

`n` is the one-based completed iteration and is checkpointed. Agent 4 estimates `N`
from the bounded preflight throughput rehearsal and freezes `N`, `n_ref`, and the
entire schedule curve before the production launch. Do not recompute the horizon from
changing production speed.

Move behavior-KL retains the accepted Phase 9 controller unless Agent 1 documents a
paper-required difference:

```text
target KL: 0.015
initial beta: 0.005
beta bounds: [1e-4, 0.2]
hard mean KL limit: 0.08
```

The setup controller is independent. Behavior KL and any other paper regularizer must
be named and logged separately; never report one as though it were the other.

## 10. Raw, behavior, and EMA identities

Raw weights generate all training data. EMA weights never act in the training
population. Every checkpoint stores both.

A paired Phase 17 checkpoint binds:

- exact Phase 9 start identity;
- raw and EMA move states;
- raw and EMA setup states;
- both optimizers and independent KL controllers;
- move and setup scheduler positions and optimizer-step counts;
- all RNG namespaces and counters;
- active-game engine states and their setup episodes;
- boundary-target carry state;
- completed-setup queue and setup-pool identity;
- run/config/source digests and elapsed active-training time.

Planned checkpoint/resume must preserve the active population exactly. If exact
active-game persistence proves impossible, Agent 4 must stop for operator review;
repeatedly discarding in-flight games would bias setup outcomes toward short games.

Evaluation exports contain paired EMA weights and a manifest. A move-only evaluator
still records the paired setup digest and marks setup use as `false`.

## 11. External evaluation contract

External evaluation runs on the operator's separate MacBook every 30 minutes. Agent 5
works with the operator conversationally and must not assume SSH, shared storage,
network direction, software installation permission, or machine capacity.

One immutable composite benchmark manifest contains two named lanes:

1. `move_only`: fixed boards/setups and fixed evaluation opponents, EMA move weights;
2. `joint_move_setup`: fixed setup RNG cases and fixed opponent cases, paired EMA
   move/setup weights.

The preferred move-only base is the accepted full `phase16_benchmark_v1`:

```text
digest: ebd130198ea500248b32df990bee876583a10d53546f38a6346ec522407320c2
```

The new composite pack receives a new digest. Both lanes report overall EWR and
opponent/setup/color strata. Historical, Phase 9, rule-based, and stress opponents
are evaluation instruments only.

Candidate times are hour 0 and every 30 minutes through hour 12: 25 candidates. Each
candidate is immutable and transferred as a partial bundle before atomic publication.
The remote side recomputes candidate, model-state, pack, config, and evaluator-source
identities. A returned receipt binds every identity, result digest, host identity, and
runtime. Never evaluate or transfer a mutable `latest` filename.

The cadence is feasible only if transfer + verification + both evaluation lanes +
receipt return has p95 below 25 minutes. Backlog, retry, or skipped cadence is explicit;
never attribute an old result to a newer timestamp.

## 12. Gates and time budgets

Only the setup-network and remote-computer gates may receive extended experiment time.

### Correctness gate — at most 30 minutes

- rules, counters, orientation, legal actions, and observation/action identities;
- exact Phase 9 start digest;
- sampled-not-argmax action selection;
- both seats use the current raw policy;
- forced rebind test on an already-running game;
- true fixed transition count, and boundary targets under the section 6
  reduction invariant (`G-M4a`; `G-M4b` retired by operator decision D2);
- checkpoint/save/load/export identity;
- structural no-search and no-training-opponent assertions.

### Setup gate — target 60–90 minutes

- at least 5,000 samples split across colors;
- zero inventory, legality, placement, or orientation failures;
- exhausted-token adversarial masking and autoregressive causality tests;
- deterministic trace under identical snapshot/seed and changed draws under changed
  seeds;
- reflection-class uniqueness, per-square entropy, pairwise distance, flag/bomb
  support, sequence entropy, and effective support;
- Red/Blue/draw outcome-sign tests and synthetic reward-flip gradient test;
- real short soak proving nonzero setup optimizer steps, gradients, and model-digest
  change from completed game outcomes;
- raw/EMA/optimizer/KL/queue checkpoint round trip;
- five-epoch setup throughput measurement.

Calibrate diversity alarms against the initial masked model and soak. Do not borrow
frozen-library family thresholds. A provisional production hard floor is 60% of the
initial mean prefix entropy for three consecutive checks and flag effective support
below four.

### External gate — conversational duration

Send one early paired candidate to the MacBook. The returned candidate identity,
model identities, pack identity, evaluator identity, and result receipt must all match.
Measure the full cadence latency before unattended operation is authorized.

### Safety wiring — bounded smoke only

Preserve relevant accepted safety features. Inject one representative stop event and
prove the supervisor records the reason and exits safely. Do not spend the phase
exhaustively testing every previously accepted safety branch.

## 13. Production stop policy

Stop immediately on:

- rules, orientation, legality, candidate, or digest mismatch;
- any decision recorded under the wrong current move-policy digest;
- nonfinite loss, gradient, parameter, or schedule value;
- setup generation/masking failure or silent fallback attempt;
- search or a non-current training opponent entering collection;
- evaluation result bound to the wrong candidate or benchmark;
- unrecoverable checkpoint/resume identity failure.

Stop on persistent collapse:

- fixed-pack EWR at least 0.15 below hour 0 for three consecutive evaluations;
- move mean KL above 0.08 for three consecutive windows, unless the existing hard
  veto stops earlier;
- setup KL above its Agent 1 hard range for three consecutive setup updates;
- setup mean prefix entropy below 60% of its initial baseline for three checks;
- flag effective support below four;
- move entropy below 25% of its first-hour median for five windows;
- no setup optimizer update for one complete 30-minute interval after warm-up while
  games and setup episodes complete;
- setup queue age/backlog crossing the frozen Agent 1 ceiling for three windows.

One noisy EWR, KL, or entropy reading produces a warning, not a stop. Other accepted
safety telemetry remains enabled but does not require a new experiment.

## 14. Checkpoint-selection rule

The last checkpoint does not win automatically. Agent 7 produces a Pareto shortlist
from eligible hour 6–12 candidates using:

1. mean composite-pack EWR;
2. worst opponent/setup stratum EWR;
3. three-point rolling-median direction from hour 6 through hour 12;
4. move-only non-regression;
5. setup legality, entropy, and diversity floors;
6. KL, queue, and training stability.

The report recommends a checkpoint and up to two alternatives. A single isolated EWR
peak cannot override a poor worst stratum or collapsed setup distribution. The operator
makes the final promotion decision.

## 15. Agents, dependencies, and handoffs

| Agent | Charter | May start | Handoff |
|---|---|---|---|
| 1 | contract, paper map, identities, baseline | now | `phase17_contract_handoff_v1` |
| 2 | fixed-transition move learner | after Agent 1 | `phase17_move_handoff_v1` |
| 3 | autoregressive setup learner | after Agent 1 | `phase17_setup_handoff_v1` |
| 4 | tandem runner, persistence, schedules | after Agents 2 and 3 | `phase17_tandem_handoff_v1` |
| 5 | conversational external evaluation | discovery after Agent 1; implementation after Agent 4 export schema | `phase17_external_eval_handoff_v1` |
| 6 | preflight and launch authorization | after Agents 2–5 | `phase17_launch_decision_v1` |
| 7 | 12-hour run and closeout | after Agent 6 GO and operator launch approval | `phase17_run_closeout_v1` |

Agents 2 and 3 may work in parallel only after Agent 1 freezes their shared schemas.
Agent 5 may conduct remote discovery concurrently, but must not freeze bundle details
until Agent 4's export schema lands. Consume other agents only through verified handoff
artifacts, never their work-in-progress state.

## 16. Additive namespaces

Use additive Phase 17 paths. Agent 1 may refine the exact module split in its handoff:

```text
stratego/training/phase17/
stratego/evaluation/phase17/
tests/training/phase17/
tests/evaluation/phase17/
scripts/*phase17*
checkpoints/phase17/
data/phase17/
reports/phase17/
```

Do not write production artifacts into Phase 9, 14, 15, or 16 directories.

## 17. Reports and claims

Each agent writes:

```text
reports/phase17/agent_0N_report.md
reports/phase17/agent_0N_*.json|jsonl|csv
reports/phase17/<declared_handoff>.json
```

Every report states what was built, exact commands/tests run, runtime, artifact
digests, failures, known limitations, and what was not established. Do not claim that
an unrun gate passed, that fixed-board EWR measures the setup policy, or that a
production failure establishes a scientific conclusion.

## 18. Later-phase boundary

Phase 17 ends with a frozen paired move/setup checkpoint recommendation. Only after
the operator promotes it may a later phase:

1. collect selected-policy trajectories;
2. train the joint autoregressive belief model separately;
3. attach belief-guided search;
4. add stochastic human-facing move selection.

No Phase 17 agent implements or quietly prepares belief-guided search.
