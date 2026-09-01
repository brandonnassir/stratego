# Phase 17 — Agent 6 report
## Short launch freeze for `RUN-2026-B`

**Decision: GO.** Explicit operator approval is still required to start the 12-hour run.

| | |
|---|---|
| work package | `phase17` |
| recipe | `phase17_simple_paper_tandem_v1` |
| run ID | `RUN-2026-B` |
| source closure digest | `4aa5c3504f775f6b6963888ea74adb6e56c4d6a2cc5da6700955cbd79c349cc5` |
| production config digest | `4e7abad83dab4a9bec1cf82f4b238059eb9de0c344bf0025f91aa0a703b2d1b7` |
| schedule digest | `2607502309d51f2c6d7ecb9796d74d1f9fab5de43dd32bb590a2e5e9337787df` |
| run digest | `c385ffeae635386add2df64ddbf259bdd8de7cf42535d1cd9eafe5ed3759779c` |
| freeze HEAD | `90278aa4fe94b2b9a661fa3d27229e4d3c8393bc` |
| manifest digest | `045e09b5ecf9dc1eb91889e5bb013e2b7a1542c65fc3020261049bf2e52e3467` |
| decision digest | `2ecc95e09c98b35d325bd43f9b52d2999fb3cd54747a8598f645093dc88e947c` |

No training was run. The complete local check was greps, digest recomputation, one
`--describe`, and one focused test module (27 s of compute) — well inside the 15-minute
cap. No tandem smoke, evaluator check, setup soak, diversity gate, EWR test, sweep,
controller calibration, queue study, schedule study, resume campaign, or
failure-injection matrix was performed.

## 1. Decision

```text
GO   the accepted Agent 4B smoke, Agent 4C's corrections, the identities and the
     launch command all match. No identity, routing, legality, numerical,
     persistence or prohibited-path check failed.
```

Three conditions travel with the GO (section 4). One of them is material and changes
what Agent 7 should expect at h12; none of them is an integrity failure.

## 2. What was checked

### 2.1 Inputs — PASS

`phase17_simple_tandem_handoff_v2` is present with `ready_for_launch_freeze: true`, a
non-empty 22-file source closure, and four distinct commits correctly separated
(evidence parent `eab8a33`, implementation `3be8bba`, correction `67b186a`, deliverables
found by grep because a file cannot name the commit containing it).

HEAD is `90278aa`, two commits past the correction commit. Neither `f3f0f14` nor
`90278aa` touches a source-closure member — they touch `reports/` and one test file — so
the authorized source digest is unchanged from `67b186a`. Every closure member is
committed and clean.

### 2.2 Agent 4B's smoke — PASS, and its recorded commit field is the known mis-attribution

The smoke artifact is intact: 9/9 checks passed, no stop predicates fired, no warnings,
291 s across 3 iterations. Its nine checks map one-to-one onto D10 §6's nine items.

Its `commit` field says `eab8a33`. That is wrong, and it is exactly the defect Agent 4C
correction 6 addressed. I established which tree actually ran it rather than taking
either claim on trust: each of the smoke's 11 recorded source hashes reproduces the blob
at `3be8bba`, and none reproduces the blob at `eab8a33`. The evidence is sound; only the
label was wrong, and the label is now corrected in the handoff.

Five of those 11 files have changed since. The important question is whether the change
reaches the recipe, and it does not:

- the **production config document built from current source is byte-identical** to the
  `production_configuration` block the smoke recorded;
- the whole setup half (`setup_contract`, `setup_episode`, `setup_learning`,
  `setup_metrics`), plus `checkpoint.py` and `export.py`, are byte-identical to the
  smoke's tree;
- `move_trainer.py`, `move_loss.py` and the `transition_*` modules are unchanged since
  `3be8bba` (they were absent from the smoke's hand-written closure — the omission 4C
  fixed — but the git delta covers them);
- a scan of the entire post-smoke delta for recipe-bearing constants (transition budget,
  epochs, PPO clip, alpha, learning rate, EMA decay, KL coefficients, argmax, search,
  belief) returned only a median helper whose lines moved, and comments.

The delta is the six Agent 4C corrections and nothing else.

### 2.3 Agent 4C's corrections against current source — PASS

`tests/training/phase17/test_agent_4c_corrections.py`: **30 passed, 0 failed, 27.36 s.**

### 2.4 Dry run — PASS with condition A6-C2

`--describe` reports `RUN-2026-B`, the source closure and its digest, the schedule
digest, the telemetry schema, 25 expected candidates, and the production command with
the digest already substituted. All digests are non-empty, and the source digest is
recomputed on every invocation rather than read from a constant — I confirmed
`source_identity.py` contains no digest literal.

I recomputed the production config digest independently and got `4e7abad8…`, matching
the handoff. It is run-ID-bound: the same builder at the smoke's run ID yields a
different digest, so a smoke config cannot be mistaken for the production one.

The dry run does **not** print the Phase 9 start identity or the fresh-setup identity,
which §2.3 of my brief asks it to report. I verified both directly instead and froze
them in the manifest (section 3 below).

### 2.5 Destinations and process state — PASS

`checkpoints/phase17/RUN-2026-B` does not exist. 198 GiB free against ~0.16 GiB of
candidates (measured 6.7 MB per export × 25) plus checkpoints. No `run_phase17`,
`phase17_training`, or `phase14_launch` process is running.

### 2.6 h0 and export cadence — PASS

`export_hour_zero` raises if `runner.iteration` is nonzero, and `main()` calls it before
the `while … session.step()` loop, so h0 cannot land after an optimizer update. The
proof that this works is in the smoke's own h0 record: its `move_ema_model_state_digest`
is `f1df694d59e3…`, **exactly the accepted Phase 9 model-state digest** — the untouched
starting policy.

Later boundaries derive from `elapsed_active_training_seconds`, never wall clock, at
1800 s, capped at 43,200 s (h12), giving ordinals 0–24. An iteration spanning two
boundaries yields both, so the cadence cannot silently shorten. Candidate names are
immutable; `write_paired_export` refuses an existing name and there is no `latest`.

### 2.7 Prohibited paths — PASS

`runner.py` contains no reference to search, belief, historical, handcrafted, evaluator,
or transport. `BELIEF_LOSS_WEIGHT` is `0.0`. Both seats are bound to
`CURRENT_POLICY_TOKEN`, and the participant ledger is a runtime proof rather than a
declaration: every acting `behavior_model_state_digest` must be one the live cell has
actually held, an unknown digest arms `I2`, and any rule/historical/search decision arms
`I5`. Agent 5's evaluator, worker, publisher, and the retired transport endpoint are all
outside the frozen closure and imported by no closure member.

One import needs naming because it looks like a breach and is not:
`stratego/belief/phase15/orientation.py` is imported for `assert_engine_orientation`, a
pure assertion that re-derives placement from the engine's own `SETUP_SQUARES`. Contract
§7 mandates it. It is not a belief model, belief loss, or belief-guided input.

## 3. Identities frozen here that the dry run does not print

**Phase 9 move start.** The file at `checkpoints/phase9/selfplay_c1_v1.pt` hashes to
`dfd698e5b6cf…`, matching the contract and the frozen constant. The loader refuses on
either the file digest or the model-state digest (`f1df694d59e3…`) before returning a
model, so a convenience copy with different bytes cannot start the run.

**Fresh setup start.** The production run ID *refuses an injected setup model outright*
rather than trusting a caller to pass a fresh one. `build_setup_model(seed=17)` yields
802,320 parameters — the contract's FF-512 figure — at state digest `9dc73986f4e3…`,
reproducible and identical on CPU and MPS. A fresh EMA at zero updates equals the raw
weights, so **h0's setup EMA digest must be `9dc73986f4e3…`**.

I checked this claim rather than asserting it. The smoke's h0 recorded a *different*
setup digest, `9568aaa529d8…`. The reason is that D10 §3 requires the smoke to use a
different seed, and it did: seed 1017 against production's 17. Rebuilding seed 1017 from
current source reproduces `9568aaa529d8…` exactly — which confirms the smoke's h0 came
from the same from-scratch construction path production will use, and that no rehearsal
setup state can leak in.

Agent 7 now has two exact h0 expectations to compare against, covering what `--describe`
omits.

**Out-of-closure dependencies.** The closure covers Phase 17's own modules and the entry
script, but the run also loads six accepted Phase 9/15/16 helpers whose bytes do not move
the source digest — an edit to `phase16/schedules.py` would change the run's behavior
silently. All six are clean at HEAD and their hashes are frozen in the manifest as a
secondary identity.

## 4. Conditions on Agent 7

### A6-C1 (material) — the 25th candidate is rate-dependent; 24 is an acceptable outcome

The loop terminates on **iteration count**, not on 12 active hours:
`while runner.iteration < config.total_iterations and not stopped`. `N = 640` was sized
as `floor(43200 / 67.404)`, and that floor is what costs the last boundary:

| source | mean s/iteration | 640 iterations | h12 boundary at 43,200 s | candidates |
|---|---|---|---|---|
| Agent 4 throughput rehearsal | 67.404 | 43,139 s = 11.983 h | **61 s short** | 24 |
| Agent 4B D10 smoke | 67.995 | 43,517 s = 12.088 h | clears by 317 s | 25 |

The outcome turns on whether production averages above or below 67.5 s/iteration, and
the two pieces of measured evidence straddle that line. Agent 7 §3 says "run until active
elapsed time reaches 12 hours"; the frozen code stops at `N`. The frozen code governs —
the move LR and entropy schedules are horizoned to this `N`, and contract §9 forbids
recomputing the horizon from production speed.

**Instruction.** Record the actual count under Agent 7 §6's "or account explicitly for
each missing ordinal". A 24-candidate run is **not** an integrity failure and must not be
treated as a stop. Do **not** extend past `N` to manufacture the 25th. If the run ends at
24, the hour 6–12 selection window ends at h11.5 with 12 candidates instead of 13.

### A6-C2 (minor) — use the manifest for the Phase 9 and fresh-setup recompute

`--describe` does not print those identities. Use the manifest's `phase9_move_start` and
`fresh_setup_start` blocks as the frozen values for the Agent 7 §1 recompute, and compare
h0's recorded move and setup EMA digests against the two `h0_expectation` values.

### A6-C3 (carried by instruction) — D10-ADV-SCALE remains uncompensated

The printed setup advantage puts `alpha*(I - h)` in nats against an outcome term bounded
by 2. The smoke measured an entropy-to-outcome magnitude ratio of **2.84** at iteration 1
(2.70 vs 0.999). D10 §4 ships this deliberately and forbids a compensating scale, floor,
centering rule, or controller. Record the component magnitudes as telemetry; do not tune.
This is an experiment result, not a defect to repair mid-run.

### A6-C4 — closed, recorded so it is not carried forward

Agent 4's per-resume telemetry-row loss is fixed by 4C correction 2. The checkpointed row
is retained exactly once and genuinely uncheckpointed later rows are still truncated. I
reran the focused proofs. No action.

## 5. What this freeze does not establish

- **No strength claim.** No benchmark lane was evaluated; that is D11's post-training work.
- The D10 smoke was **not** rerun at production shape, by instruction.
- Resume was **not** exercised on MPS. MPS is not bitwise reproducible run to run, so
  Agent 4C's resume proofs are CPU. The production run is MPS.
- Setup diversity or concentration behavior over 12 hours is unknown and, under D10, is
  telemetry rather than a gate.
- Whether the run yields 24 or 25 candidates (A6-C1).

## 6. Launch record

```text
reports/phase17/phase17_launch_decision_v2.json
reports/phase17/phase17_launch_manifest_v2.json
reports/phase17/agent_06_report.md
```

The production command, frozen with its source digest:

```bash
nohup caffeinate -dimsu .venv/bin/python scripts/run_phase17_training.py --run-id RUN-2026-B --start --i-am-agent-7 --source-digest 4aa5c3504f775f6b6963888ea74adb6e56c4d6a2cc5da6700955cbd79c349cc5 > <log> 2>&1 &
```

It refuses to run without `--i-am-agent-7` (exit 2) and without a matching
`--source-digest` (exit 3, naming both digests). Use `.venv/bin/python`, not bare
`python` — the pyenv shim has no torch.

Agent 5 evaluation begins **only** after Agent 7 freezes the run and its candidate
ordinals. No evaluator runs while `RUN-2026-B` is training.

Explicit operator approval is still required to start the 12-hour run.
