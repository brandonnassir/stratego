# Phase 17 — Operator Decision D10
## Simplified paper-shaped tandem self-play is now the active experiment

_Written 2026-08-28 after Agent 4 completed the tandem infrastructure._

## 1. Decision and precedence

The project will stop treating the setup learner as an independently certified
subsystem. The move policy and setup policy are one evolving self-play system and will
train together immediately.

This decision supersedes the active effect of D3, D4, D5, D7-B, and D9-B wherever
they conflict with the recipe below. Their reports and measurements remain immutable
historical evidence. In particular, the following are retired from the active run:

- the adaptive setup-KL controller and all target/beta-bound logic;
- the endpoint-re-horizoned setup entropy schedule;
- equation `phase17_setup_update_v2` and its invented uncentered normalized bonus;
- the standalone setup-diversity pass/fail gate;
- the fixed setup-episode quota, warm-up gate, and age-calibration campaign; and
- broad preflight certification beyond checks that could silently falsify the run.

## 2. What Agent 4 established and what is retained

Agent 4 completed the tandem foundation at commits:

```text
a4aa12d  Implement Phase 17 Agent 4 tandem runner, persistence and guards
c2c0365  Reconcile the two tandem draw-rate measurements in the Agent 4 report
```

Retain and reuse:

- exact Phase 9 move start and fresh move optimizer/schedule/EMA initialization;
- fixed-transition move collection and unfinished-game bootstrapping;
- sampled legal moves from the current raw move policy for both seats;
- current-policy rebind for active games;
- fresh autoregressive setup generation and outcome binding;
- paired raw/EMA checkpointing, exact active-game persistence, exports, telemetry,
  and integrity-oriented safety stops;
- no-search/no-belief/no-training-opponent boundaries; and
- the external 30-minute evaluation interface.

Agent 4 also established that the tandem signal is usable: its 200-iteration soak
completed 32,043 games with 1.87% draws, compared with 83.3% draws in Agent 3's
uniform-random fixture. Setup work was only about 3.6% of measured iteration time.

Agent 4's adaptive beta sat at its ceiling for 97.5% of the soak. That is evidence for
removing the controller, not for another controller-calibration experiment.

Agent 4's work is complete. Its session was terminated on 2026-08-28 after several
wait shells became stuck by matching their own `pgrep` pattern. No Phase 17 training
process was active, no Agent 4 work was lost, and the only remaining working-tree
changes were the three older documentation modifications already identified as
unrelated.

## 3. New run identity and initialization

The simplified experiment is a new lineage:

```text
work package: phase17
recipe: phase17_simple_paper_tandem_v1
run ID: RUN-2026-B
integration base: c2c0365
```

At production start:

- load only the accepted Phase 9 move weights from
  `checkpoints/phase9/selfplay_c1_v1.pt` with file and model-state digest checks;
- create fresh move optimizer, move schedule state, move KL state, and move EMA;
- initialize the setup network from scratch under a recorded seed;
- create fresh setup optimizer and setup EMA state;
- load no setup weights, setup optimizer state, setup pool, setup queue, or setup EMA
  from any Agent 3/4 rehearsal; and
- create h0 before either learner updates.

A smoke/preflight run uses a different run ID and seed and is discarded. Production
must reinitialize from Phase 9 plus a newly random setup model.

## 4. Active tandem recipe

### Move learner

- 100% current raw-policy self-play for both seats;
- categorical sampling over legal moves, never argmax;
- exactly 65,536 learner transitions per iteration;
- unfinished traces bootstrapped at the fixed-transition boundary;
- one move epoch per iteration;
- existing Phase 17 move schedule and move KL behavior remain unchanged;
- raw weights generate training data and EMA `0.999` weights are evaluation only.

### Setup learner

```text
architecture:           existing 4-block, width-128, 4-head, FF-512 causal model
initialization:         from scratch
pool:                   512 fresh samples per side from the current raw setup snapshot
pool cadence:           regenerate at every shared/global tandem iteration
optimizer:              Adam
learning rate:          constant 5e-5
setup epochs:           5
PPO clip:               0.2
value-loss weight:      0.5
conditional-h loss:     weight 1.0; target I/10
gradient clip:          0.5
EMA decay:              0.999
behavior KL:            reverse D_KL(current || behavior)
behavior KL coefficient: fixed 0.1
entropy temperature:    alpha(n) = 0.1 * n^-0.3
iteration n:            the same one-based global tandem iteration used by the run
```

Use the paper's printed setup advantage directly:

```text
delta = (outcome - E[behavior W/D/L value]) + alpha(n) * (I - h_behavior)
```

`I` is the realized suffix information `-log p_behavior(suffix | prefix)` in nats,
and `h_behavior` is the recorded conditional-entropy prediction belonging to the raw
setup snapshot that generated the episode. Preserve the paper's separately normalized
conditional-entropy loss target `I/10`. This deliberately retires the locally invented
D7-B advantage. Record the component magnitudes; do not add a compensating scale,
floor, centering rule, horizon map, or controller.

Train once on every setup episode whose game completed during the fixed-transition
iteration, with both sides represented, then remove those episodes from the pending
buffer. Use all available completed episodes exactly once for five epochs. If none
complete, record a skipped setup update. Do not impose a fixed setup quota, wait for a
warm-up count, retain completed episodes for later balancing, or silently drop them.
The completed-episode buffer is persisted only so a crash cannot lose or duplicate an
outcome before the iteration closes.

## 5. Training population boundaries

Training contains only the current move policy and current setup policy:

- no historical checkpoints as opponents;
- no handcrafted, rule, or stress opponents;
- no search;
- no belief-model loss or belief-guided input;
- no frozen setup library or fallback; and
- no argmax move collection.

Historical and handcrafted opponents remain evaluation-only.

## 6. Minimal preflight

The complete preflight must fit within 30 minutes and should normally be much shorter.
Run one short end-to-end tandem smoke using the production code path. Check only:

1. exact Phase 9 move identity and fresh setup identity;
2. both move seats use the current raw policy and sample legal actions;
3. fresh legal, inventory-correct, correctly oriented setups come from the current raw
   setup model with no library fallback;
4. exactly the configured move-transition count is emitted;
5. at least one real completed game updates the setup model for five epochs;
6. setup reverse KL has fixed coefficient `0.1`, with no adaptive controller update;
7. setup alpha equals `0.1 * n^-0.3` at the shared global iteration;
8. a paired checkpoint saves and reloads without identity loss or duplicated/dropped
   completed setup outcomes; and
9. search, belief, historical, and handcrafted training participants are absent.

Do not run another 5,000-setup gate, standalone setup soak, entropy-floor experiment,
controller calibration, population sweep, queue-arrival study, strength test, or broad
failure-injection campaign. Existing Agent 2/4 evidence covers the retained machinery.

## 7. Monitoring and stops

Stop only for conditions that make results invalid or unrecoverable:

- wrong checkpoint/model/config/benchmark identity;
- wrong policy routed to either training seat;
- illegal action, illegal/misoriented setup, or silent setup fallback;
- nonfinite loss, gradient, or parameter;
- search, belief, historical, or handcrafted participant entering training;
- fixed-transition count violation;
- checkpoint corruption or loss/duplication of setup outcomes across resume; or
- inability to continue safely because of resource exhaustion.

EWR decline, high but finite KL, setup entropy decline, low diversity, game-length
change, and setup concentration are telemetry and warnings, not automatic stops. The
12-hour learning curve is the experiment. Do not tune or restart mid-run because the
curve looks bad.

## 8. External evaluation and closeout

Preserve paired EMA evaluation every 30 active minutes using immutable candidate
digests and the fixed benchmark pack. Perform one h0 external identity round trip so
results cannot be silently attached to the wrong model or pack; do not turn this into
a broader model-quality gate.

After 12 active hours, select from hour 6–12 using learning-curve direction, mean EWR,
and robustness across the fixed strata. Setup entropy/diversity informs interpretation
but no longer makes a checkpoint ineligible by itself.

If this Phase 9 plus fresh-setup experiment behaves badly, preserve it as the result.
The next candidate experiment is a separately versioned tabula-rasa run in which both
move and setup networks start from scratch.
