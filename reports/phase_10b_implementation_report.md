# Phase 10B Implementation Report

## Status: PAUSED — INCOMPLETE

Phase 10B is the **optional** setup-conditioned self-play fine-tuning experiment described in
`instructions/phase_10_sequential_agent_plan/OPTIONAL_PHASE_10B_SETUP_CONDITIONED_FINE_TUNING_AGENT.md`.
It is advisory only. It does not reopen Phase 10, does not alter the accepted Phase 9 move
model, and has neither blocked nor modified Phase 11.

The run was **paused by operator request** immediately after the iteration-5 validation pass,
at a clean resumable boundary. **There is no Phase 10B classification.** Checkpoint selection,
the sealed final evaluation and Gates A–H have not run, and the sealed test bank has never
been opened for outcome evaluation.

```text
completed      5 of 30 iterations       10,240 of 61,440 training games
optimizer      9,074 steps              10 of 60 optimizer epochs
wall clock     2.00 h of the 12 h ceiling
validation     1 of 6 scheduled passes (iteration 5)
final test     never opened
classification none — the experiment is unfinished, not failed
```

---

## 1. Repository revision and environment

```text
starting revision   17188a5   Implement Phase 10 Agent 7 sealed final test and formal Phase 10 closure
final revision      6761308   Implement Phase 10b setup-conditioned fine-tuning and Phase 11 Agent 1 contract
platform            macOS-26.5.2-arm64
python              3.13.2
torch               2.13.0, MPS available
collection device   mps, inference batch shape 64, 96 games in flight
training device     mps, float32, AdamW
evaluation device   cpu, 8 worker processes (measured ~5x faster than MPS for
                    single_request greedy forwards, which are latency-bound)
```

---

## 2. Frozen upstream identities — all verified from live bytes

Every identity below was recomputed from the live files before the first rollout existed, and
re-proved at every stage boundary during the run.

```text
accepted Phase 9 checkpoint   checkpoints/phase9/selfplay_c1_v1.pt
  SHA-256                     dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
  model-state digest          f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
  parameters                  863,959
  C1 config digest            31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d

frozen selector               P10-D | model_T | T=0.75 | 0.35 neutral / 0.65 learned
  selector config SHA-256     6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668
  model_T coefficient digest  d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc
  trait scaler digest         fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9
  phase10_system_v1 digest    615cc3c3a4fab6e4400e20a5a93b13a08c43ab6c3ca63828c6a64742e98175d2

Phase 7 library               7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
```

The accepted Phase 9 checkpoint is **byte-identical** before and after every action this run
took. `phase10b_checkpoint.assert_phase9_untouched` re-hashes it at each stage boundary and
after every iteration; it never diverged.

---

## 3. Phase 10B contract digests

```text
phase10b_contract_v1            7159966745199a1bf1c2f5b9f2bcf75e30cc4cd40af89b7de0f2a685eba0c5c0
phase10b_seed_v1                cba971802f393169ec9543ec8708c973f7fb5fde4c03e74251bf3c5490117196
phase10b_population_v1          a51775d85c37eb3c573d934c8453c48d82b8928bb4cf1dda39f33b127af1e658
phase10b_validation_bank_v1     9295c2efc8467dced84ec32d0903d0e44f8b9f0819e908daa39812903be9a1a2  (256 cases)
phase10b_test_bank_v1           c545abd9630286380a28c3e6eeb4c89be301e91b9f04b71b9f47d55189680322  (512 cases, SEALED)
```

### Root seeds and derived domains

```text
master 20260819021   rollout schedule 20260819022   opponent selection 20260819023
setup selection 20260819024   training order 20260819025   validation schedule 20260819026
validation bootstrap 20260819027   final bootstrap 20260819028
```

Ten domains are declared: `rollout_game`, `opponent_bucket`, `opponent_identity`, `red_setup`,
`blue_setup`, `action_sampling`, `training_order`, `archive_selection`, `validation_case`,
`bootstrap`. Nine are consumed by a real draw. `opponent_bucket` is **reserved with a recorded
reason**: the population bucket and the opponent-bucket policy are assigned by contiguous
ordinal subranges, so both are exact by construction and nothing is drawn. Recording a reserved
domain rather than deleting it keeps any future bucket-level draw from borrowing another
stream. See `phase10b_seed.seed_contract()`.

Derivation is `blake2b(payload, digest_size=8, person='strat10b') >> 1` over
`"<rollout version>:<domain>:<domain root>:<identity parts>"`. No seed reads worker count,
process id, path, wall clock or arrival order; `phase10b_seed` performs no I/O at all.

---

## 4. Population and setup conditioning

Every training game draws **both** sides through the frozen P10-D selector on the Phase 7
train split, from two separate seed domains, so Red's draw is not a function of Blue's. The
selector receives only own colour, requested split and its own selector seed — it never
receives the opponent's setup, hidden rank truth, an outcome prediction, checkpoint strength or
matchup identity.

Exact per-iteration counts (largest-remainder allocation of the plan's percentages, ties broken
by declared order; both derivations are re-checked at import):

```text
current  (10B learner self-play)   1,229    60%
anchor   (accepted Phase 9)          409    20%
archive  (10B history)               205    10%
opponent (rule / stress)             205    10%
                                   -----
                                   2,048
```

Opponent bucket: strategic 62, tactical 51, basic 21, information_miser 21, scout_rush 20,
miner_rush 20, random 10.

**Declared consequence of the archive rule.** The plan's active history is "the accepted Phase 9
anchor + up to 4 most recent Phase 10B archives", sampled uniformly, anchor never evicted.
Iterations 1–5 have produced no Phase 10B archive, so the archive bucket's pool is exactly the
anchor and the anchor carries the whole 30% checkpoint-opponent share until `A005` exists. This
was declared in the contract before the run, not discovered afterwards. From iteration 6 the
window is `['P9A', 'A005']`.

---

## 5. Training schedule

Phase 10B inherits the accepted Phase 9 PPO/KL machinery unchanged — `phase9_batch_loss` is the
objective, `KLController` is the controller, and the advantage construction, filter,
standardization and WDL/belief targets are reached through `phase9_targets`. The plan freezes
exactly two departures, and only these two were made:

```text
learning rate    0.25x -> 0.10x of the accepted Phase 9 canonical start (3e-4, candidate P9-C)
                 7.500e-05 -> 3.000e-05, linear across 30 iterations
                 (replaces Phase 9's "constant, no warmup, no decay")
entropy          0.0010 -> 0.0005, linear across 30 iterations
```

Everything else is the accepted Phase 9 value: PPO clip 0.20, behaviour-KL target 0.015, hard
KL veto 0.08, hard clip-fraction veto 0.75, gradient clip 1.0, AdamW with weight decay 0.01,
float32, minibatch 512, 2 epochs per rollout, no replay, advantage lambda 0.5, WDL lambda 0.8,
advantage filter `tau = max(Q75(|A|), 0.01)`, standardized selected advantages. **No search was
used in training.**

---

## 6. Iterations completed

| it | games | learner decisions | steps | LR | entropy | KL e0 | KL e1 | clip e0 | clip e1 | retention | collect s | train s |
|----|-------|-------------------|-------|-----------|---------|----------|----------|--------|--------|------|-----|------|
| 1 | 2,048 | 460,424 | 1,800 | 7.500e-05 | 0.00100 | 0.009689 | 0.012896 | 0.1315 | 0.1649 | 0.250 | 318 | 1008 |
| 2 | 2,048 | 466,477 | 1,824 | 7.345e-05 | 0.00098 | 0.009863 | 0.011923 | 0.1333 | 0.1569 | 0.250 | 321 | 1060 |
| 3 | 2,048 | 481,263 | 1,880 | 7.190e-05 | 0.00097 | 0.010061 | 0.012762 | 0.1315 | 0.1621 | 0.250 | 340 | 1159 |
| 4 | 2,048 | 460,309 | 1,800 | 7.034e-05 | 0.00095 | 0.010049 | 0.013315 | 0.1301 | 0.1612 | 0.250 | 352 | 1064 |
| 5 | 2,048 | 453,098 | 1,770 | 6.879e-05 | 0.00093 | 0.010824 | 0.013464 | 0.1360 | 0.1605 | 0.250 | 314 | 1010 |

Behaviour KL sits just under the 0.015 target throughout and never approaches the 0.08 hard
veto; the adaptive controller never needed to move beta. Clip fraction is stable near 0.13–0.17
against a 0.75 veto. Advantage retention is 0.250 at every iteration, which is the Q75 filter's
design point rather than a coincidence.

### Training safety counters — all zero across all five iterations

```text
kl_hard_limit_breaches            0     non_finite_losses          0
clip_fraction_hard_limit_breaches 0     non_finite_gradients       0
illegal_targets                   0     non_finite_parameters      0
data_mismatches                   0     checkpoint_errors          0
behavior_identity_mismatches      0     rollout_identity_mismatches 0
```

---

## 7. Validation pass at iteration 5 (the only one completed)

256 logical paired cases, 2 colour-swapped games per case, 6,144 games, 211 s. Both arms read
the same frozen selector seeds and draw the same own-side arrangements — the **only** difference
between arms is the move policy. Paired-unit percentile bootstrap, 10,000 replicates, 95%,
logical case as the resampling unit.

```text
S10B = +0.02959   ELIGIBLE

                       candidate   Phase 9    delta      95% CI
direct  (P10-D)          0.5811       —      +0.0811   [+0.0459, +0.1162]
neutral (rollback)       0.5059       —      +0.0059   [-0.0312, +0.0410]
strategic                0.8291     0.8506   -0.0215   [-0.0615, +0.0176]
tactical                 0.8398     0.8301   +0.0098   [-0.0244, +0.0439]
phase 8 anchor           0.8379     0.8604   -0.0225   [-0.0586, +0.0137]

guards                 candidate   Phase 9
random                   0.9941     0.9863    (red 0.992 / blue 0.996)   >= 0.95 PASS
basic                    0.8809     0.8877    (red 0.902 / blue 0.859)   >= 0.80 PASS
neutral rollback         0.5059                                          >= 0.48 PASS
```

### What this early read does and does not say

The **direct** comparison is the primary scientific question, and at iteration 5 it is clearly
positive: the fine-tuned policy scores 0.581 against the accepted Phase 9 policy when both play
under P10-D, with a paired lower bound of 0.546. Setup-conditioned adaptation is happening and
it is measurable well outside noise.

The **breadth** picture is less comfortable. Against Strategic and the Phase 8 anchor the
candidate is *behind* the accepted Phase 9 policy by about two points each, and the neutral
rollback margin is barely positive. This is the specialization-versus-breadth tradeoff the
plan's gates exist to detect.

**Diagnostic only — this is not a gate evaluation.** Gates A–H are defined on the sealed
512-case test bank, which has never been opened, and gate booleans may only be computed there.
If the iteration-5 validation deltas were to hold unchanged onto the test bank, Gate A
(direct adaptation) would pass comfortably, while Gate C (strong-opponent composite,
`Delta_L = -0.0107`, requires point >= 0.00) and Gate D (individual regression guards; Strategic
LB -0.0615 and Phase 8 LB -0.0586 against a > -0.03 floor) would not. Gate B's neutral lower
bound (0.4688) also sits just under its 0.47 floor. These are 256-case intervals at iteration 5
of 30, with the learning rate still near its maximum and 25 iterations of decay remaining — they
are an early reading of a trajectory, not a verdict on the experiment.

---

## 8. Artifacts

```text
reports/phase_10b_data/
  agent_10b_contract.json             frozen contract, seeds, population, banks, trainer semantics
  agent_10b_run_state.json            complete resume state and progress against the budget
  agent_10b_iteration_metrics.csv     per-iteration training metrics
  agent_10b_validation_results.csv    per-validation-pass headline numbers
  agent_10b_validation_detail.json    every delta with its full bootstrap interval

checkpoints/phase10b/
  stages/verify.json  stages/freeze.json       stage records
  run_journal.json                             the resumable run journal
  resume.pt                                    trainer state after iteration 5
  behavior_B001..B006.pt                       per-iteration behaviour snapshots
  archive/A005.pt                              first Phase 10B archive member
  validation_it005.pt                          the iteration-5 validation checkpoint
  exports/                                     evaluation-format weight exports
  cells/                                       cached per-unit evaluation rows
  run.log                                      the run log

/Volumes/Brandon_Washington/stratego_phase10b/rollouts/phase10b/
  iteration_001..005                           sealed rollout stores (~85 MB per iteration)
  iteration_006                                partial, interrupted mid-collection
```

Not yet written, because the stages that produce them have not run:
`agent_10b_training_manifest.json`, `agent_10b_selected_checkpoint.json`,
`agent_10b_final_results.csv`, `agent_10b_belief_preservation.json`, `agent_10b_acceptance.json`.

---

## 9. How to resume — read this first

### 9.1 REQUIRED: restore the rollout-store seam before anything else

Phase 10B reuses the accepted Phase 9 crash-safe rollout store rather than duplicating its
commit/recovery logic. Doing so needs **three additive optional parameters** in
`stratego/training/phase9_rollout_store.py`:

```text
validate_rollout_metadata(..., *, id_parser=None)          defaults to the Phase 9 parser
Phase9RolloutWriter(..., metadata_validator=None)          defaults to the Phase 9 validator
seal_iteration(..., scheduled_game_ids=None,               defaults to the Phase 9 schedule
                    metadata_validator=None)
```

All three default to exactly the previous Phase 9 behaviour; `tests/training/test_phase9_agent03_artifacts.py`
passes 26/26 with them applied.

**This seam is now committed in `6761308` and is no longer at risk.** It is called out here
because it was destroyed once while uncommitted: the commit that created `17188a5` reverted it,
and the first Phase 10B launch died at the first `write_game` with
`TypeError: unexpected keyword argument 'metadata_validator'`. If a future rebase, revert or
cherry-pick drops it, resume fails the same way. Verify it is present before resuming:

```bash
grep -c "metadata_validator" stratego/training/phase9_rollout_store.py
```

Expect `6` (signature, comment, assignment, use in `write_game`, `seal_iteration` parameter, and
its local binding). If it returns `0`, the seam is gone and resume **will** fail at the first
committed game; restore it from `6761308` before continuing.

### 9.2 Resume the run

The training stage is resumable and reads `checkpoints/phase10b/run_journal.json`. Iteration 6
is partially collected; only committed games survive reconciliation and the remainder is
regenerated deterministically.

```bash
.venv/bin/python scripts/run_phase10b.py --from-stage train
```

That continues at iteration 6 and then runs `select`, `final`, `gates`, `artifacts` and
`report` to completion.

### 9.3 Budget already consumed

```text
iterations   5 of 30          games       10,240 of 61,440
epochs      10 of 60          wall clock  2.00 h of 12 h  (10.0 h remaining)
```

At the measured ~23.6 min per iteration, the remaining 25 iterations need ~9.8 h plus ~0.4 h of
validation — which is *just* inside the remaining ceiling with no margin. A bounded stop before
iteration 30 is likely. **The budget must not be extended to avoid it**; the plan forbids
extending because results are weak, and an incomplete run is reported as incomplete. Validation
passes at 5/10/15/20/25 are enough for the frozen selection rule.

The 12-hour ceiling is cumulative across resumes: `run_journal.json` carries
`wall_clock.run_seconds` forward, so a resumed run continues counting rather than restarting
the clock.

### 9.4 What must not change on resume

The contract, seeds, banks, population mix, selector, LR schedule, entropy schedule and PPO
thresholds are frozen and stamped into every sealed rollout sidecar and every checkpoint.
Changing any of them invalidates the five iterations already committed. If any of the digests in
section 3 no longer reproduce, stop `BLOCKED` rather than continuing.

---

## 10. Deviations and judgement calls, recorded

1. **Additive seam in an accepted Phase 9 file** (section 9.1), committed in `6761308`. Gate H
   requires byte-identical upstream *artifacts* — checkpoints, selector config, utility, scaler,
   library — and all of those are untouched. `phase9_rollout_store.py` is source, not an artifact, and the change is
   default-preserving. Recorded here rather than left implicit.
2. **Phase 10B has its own rollout-id scheme and seed roots**, so a Phase 10B game can never be
   mistaken for, or mixed into, an accepted Phase 9 rollout. This is why the seam is needed at
   all: the Phase 9 sidecar validator would otherwise reject a Phase 10B id.
3. **`opponent_bucket` is a reserved domain with no consumer** (section 3), because the
   assignment it would govern is exact by construction.
4. **Evaluation runs on CPU, collection and training on MPS.** Both arms always use the same
   device, so no paired comparison is contaminated; the choice is throughput only, and greedy
   single-request forwards measured ~5x faster on CPU.
5. **The Phase 9 baseline arm is computed once per bank and reused** across validation passes.
   The accepted Phase 9 checkpoint does not change during Phase 10B, so these are identical
   logical games; this is a cache, not a statistical shortcut.
6. **Archive pool before iteration 5** falls back to the anchor (section 4), declared in the
   contract in advance.

---

## 11. Phase 11 statement

Phase 10B has not blocked, paused, delayed or modified Phase 11 in any way. No Phase 11
artifact was read, written or consulted. Phase 11 Agent 1 and Agent 2 work in fact proceeded
concurrently with this run and is present in the same tree, which is direct evidence rather than
an assertion. No Phase 11 or Phase 12 evidence entered training. The
accepted Phase 9 checkpoint, the P10-D selector, the Phase 10 utility and scaler, and the
Phase 7 library are all byte-identical to their state before Phase 10B began.

## 12. Handoff

```text
Phase 10B classification   none — PAUSED, INCOMPLETE (5 of 30 iterations)
selected checkpoint        none — selection has not run
direct P10-D result        validation only, iteration 5: EWR 0.5811, 95% CI [0.5459, 0.6162]
neutral rollback result    validation only, iteration 5: EWR 0.5059, 95% CI [0.4688, 0.5410]
strong-opponent composite  validation only, iteration 5: Delta_L = -0.0107
belief preservation        not measured — Gate G has not run
training safety            all counters zero across 5 iterations
budget                     10,240 games / 5 iterations / 2.00 h of 12 h
upstream preservation      all exact
```

No production replacement decision has been made or is implied. Whether this experiment is
resumed at all remains the reviewing chat's call.
