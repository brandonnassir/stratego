# Phase 17 — Agent 7 report
## Twelve-hour training operation and candidate freeze — `RUN-2026-B`

**Status: complete. `ready_for_post_training_evaluation: true`.**
Training facts only. No strength claim and no checkpoint recommendation is made here.

| | |
|---|---|
| Run | `RUN-2026-B`, recipe `phase17_simple_paper_tandem_v1` |
| Launched | 2026-08-29T14:02:33Z |
| Terminated | 2026-08-30T03:11:16Z, by operator authorization |
| Active training time | **45,569.596 s = 12.6582 h** |
| Iterations completed | **535** of the frozen 640 |
| Candidates | **25**, ordinals 0–24, all verified |
| Stop predicates | **none** |
| Warnings (P1–P7) | **none** |
| Integrity failure | **no** |

---

## 1. Prelaunch verification

All Section 1 checks passed before h0 was created. No learner or evaluator owned the
machine; the production run directory did not exist; 204 GiB were free; the host was on
AC power with sleep and display-sleep disabled and the run was wrapped in
`caffeinate -dimsu`.

Recomputed identities, each an exact match to `phase17_launch_manifest_v2`:

| Identity | Value | Match |
|---|---|---|
| Source closure | `4aa5c3504f775f6b…c349cc5`, 22 files | yes |
| Production config | `4e7abad83dab4a9b…a703b2d1b7` | yes |
| Schedule | `2607502309d51f2c…e9337787df` | yes |
| Run digest | `c385ffeae635386a…ed3759779c` | yes |
| Launch manifest digest | `045e09b5ecf9dc11…52e3467` | recomputed |
| Launch decision digest | `2ecc95e09c98b35d…3cc88e947c` | recomputed |
| Phase 9 start file | `dfd698e5b6cf536a…6179b10ea` | yes |

The dry run reported `RUN-2026-B`, `N=640`, `n_ref=80`, a 65,536-transition budget,
population 256, five setup epochs, fixed reverse KL 0.1, `alpha = 0.1·n^-0.3`,
25 expected candidates and `started: false`, and its printed production command was
byte-identical to the authorized one apart from the log path.

**Condition A6-C2 discharged.** The dry run does not print the Phase 9 or fresh-setup
identity, so both were recomputed directly using the export module's own digest walk:
the move model returned `f1df694d59e3…dffcefd` at 863,959 parameters and
`build_setup_model(seed=17)` returned `9dc73986f4e3…c1e5207aa` at 802,320 parameters.
Both equal the manifest's `h0_expectation`, and h0 was later published carrying exactly
these values.

Prohibited participants were absent from the whole 22-file closure: `evaluator` and
`transport` appear only inside docstrings, `historical` and `handcrafted` only in
refusal lists and refusal messages, there is no search or MCTS reference, and
`BELIEF_LOSS_WEIGHT` is `0.0`. The supervisor is `phase17_run_supervisor_v3`, arming
I1–I8 as immediate stops and P1–P7 as warnings that can never stop a run.

---

## 2. The run

535 iterations, numbered contiguously 1–535 with no gaps, over 12.6582 active hours.

| Quantity | Value |
|---|---|
| Mean seconds per iteration | 85.177 |
| Iteration seconds min / median / max | 63.5 / 85.0 / 107.2 |
| Transitions harvested | 35,061,760 |
| Games completed | 233,936 |
| Setup episodes consumed | 467,872 |
| Move optimizer steps | 68,480 |
| Setup optimizer steps | 37,875 |
| Memory high water | 20,488 MiB of 48 GiB |
| Terminal results | 111,099 blue / 111,820 red / 11,017 draw (4.7 %) |
| Terminal reasons | 214,242 flag capture, 10,965 move-limit draw, 8,677 no-legal-move, 52 mutual no-legal-move |

Every contract invariant held on **all 535 rows**: the 65,536-transition budget never
varied, the participant ledger held every iteration with both seats bound to
`phase17_current_raw_move_v1|P17RAW`, and historical participants, search participants,
rule-or-stress decisions, unknown model states, non-finite gradients, setup legality
failures, setup orientation failures and setup fallback attempts were **all zero
throughout**. The setup half ran exactly five epochs at a fixed reverse-KL coefficient
of 0.1 every iteration and was never skipped.

Setup-episode consumption reconciles exactly: cumulative enqueued equals cumulative
consumed at 467,872, buffer depth never exceeded 0, nothing was rejected, and against
233,936 completed games that is **exactly 2.000000 episodes per game** — both sides,
each consumed once, in its own iteration.

### Iteration cost rose, and why

Iteration cost grew from 68.3 s to a median of 85.0 s. The growth was entirely in
collection (37.9 s → ~73 s); move optimization stayed flat near 28 s and the setup half
stayed near 3.5 s. The cause is the arrival-rate effect Agent 4 recorded as A4-CF4: mean
game length fell from about 270 plies to about 145, so a *fixed* transition budget
completes far more games per iteration, and every completed game costs a fresh setup
sample. This is why the run averaged 85.177 s per iteration against Agent 4's 67.404 s
estimate.

**Condition A6-C1 resolved: 25 candidates.** A6-C1 allowed 24 or 25 depending on whether
production averaged above or below 67.5 s per iteration. It averaged 85.177 s, well
above both that threshold and Agent 4B's 67.995 s smoke rate, so the 43,200 s export
horizon was reached comfortably, at iteration 510, with 130 iterations of the frozen
horizon still unrun. The condition anticipated a possible 61-second shortfall; the
opposite occurred. No count below 24 needs explaining.

---

## 3. Candidates

25 immutable paired EMA exports, ordinals 0–24, contiguous from zero with none missing.
All 25 move-EMA digests are distinct, so no candidate silently duplicates another.
Each was verified the moment it was written and re-verified after termination; **all 25
are byte-identical across those two readings**, which is the immutability proof — nothing
was overwritten, renamed or replaced.

| Ord | Hour | Iter | Active s | Over | Move EMA | Setup EMA | Verify |
|---|---|---|---|---|---|---|---|
| 0 | h0 | 0 | 0.0 | +0.0 | `f1df694d59e3…` | `9dc73986f4e3…` | OK |
| 1 | h0.5 | 26 | 1844.7 | +44.7 | `bef41c235a6c…` | `c3f6eaff058f…` | OK |
| 2 | h1 | 46 | 3635.0 | +35.0 | `e103bf6dcc98…` | `bb8203cece34…` | OK |
| 3 | h1.5 | 66 | 5470.8 | +70.8 | `e26b7e987fb6…` | `041c6db216c5…` | OK |
| 4 | h2 | 85 | 7205.1 | +5.1 | `4fc524680c9d…` | `4383109f220a…` | OK |
| 5 | h2.5 | 106 | 9084.3 | +84.3 | `dce4c62ec19b…` | `43c3c6a25229…` | OK |
| 6 | h3 | 126 | 10875.4 | +75.4 | `69a54bf950a5…` | `677bf716c2fa…` | OK |
| 7 | h3.5 | 148 | 12607.1 | +7.1 | `e68e7b19f504…` | `3b9578619216…` | OK |
| 8 | h4 | 173 | 14485.7 | +85.7 | `3778917a14ca…` | `8772417d37f0…` | OK |
| 9 | h4.5 | 195 | 16213.0 | +13.0 | `ae0c067f1f4c…` | `a5923c602539…` | OK |
| 10 | h5 | 217 | 18032.7 | +32.7 | `cb3004db91e8…` | `5ade873ebe0e…` | OK |
| 11 | h5.5 | 239 | 19845.8 | +45.8 | `f82ca6883035…` | `800c5e4ed007…` | OK |
| 12 | h6 | 261 | 21681.3 | +81.3 | `ba55815d7ac8…` | `ede1ed43fe21…` | OK |
| 13 | h6.5 | 281 | 23478.9 | +78.9 | `6673def5c183…` | `e3f7206e62eb…` | OK |
| 14 | h7 | 302 | 25260.2 | +60.2 | `024c6847ed0b…` | `95ab3ebb909e…` | OK |
| 15 | h7.5 | 323 | 27044.4 | +44.4 | `a99cb70dee3f…` | `99a0b9559b9c…` | OK |
| 16 | h8 | 344 | 28865.1 | +65.1 | `ac7ca7ca8c5d…` | `d45b7541f141…` | OK |
| 17 | h8.5 | 363 | 30616.8 | +16.8 | `5c043cb7bc03…` | `24544b27035c…` | OK |
| 18 | h9 | 382 | 32488.5 | +88.5 | `4cd735810bed…` | `166ed2e1203f…` | OK |
| 19 | h9.5 | 403 | 34269.7 | +69.7 | `19afb14dfd73…` | `7a8ce4be03da…` | OK |
| 20 | h10 | 425 | 36035.6 | +35.6 | `97bd51751dc7…` | `392b8b08c50e…` | OK |
| 21 | h10.5 | 447 | 37856.5 | +56.5 | `45ceb2904db8…` | `81234be8b160…` | OK |
| 22 | h11 | 468 | 39636.7 | +36.7 | `8dffd9381f86…` | `748cb3ca6a0b…` | OK |
| 23 | h11.5 | 490 | 41473.1 | +73.1 | `440fedf6c03b…` | `04b223397957…` | OK |
| 24 | h12 | 510 | 43250.4 | +50.4 | `d245c704dfb7…` | `0142980d1066…` | OK |
Every candidate carries the authorized source digest `4aa5c350…c349cc5` and config
digest `4e7abad8…a703b2d1b7`, matching parameter counts of 863,959 and 802,320, a
`nominal_boundary_seconds` equal to its ordinal times 1800, and an active time at or
after its own boundary (overshoot 0.0–88.5 s). h0 was written before either optimizer
update and was never scored.

**Candidate 24 crossed the horizon**: 43,250.427 active seconds against the 43,200 s
boundary, at iteration 510. The last ordinal is **h12**, not h11.5. No candidate was
created during termination, and none was owed — the next boundary would have been
46,800 s, past the horizon.

---

## 4. Termination

Operator-authorized termination after h12, superseding the frozen 640-iteration horizon.

- **Process resolution.** Both processes were positively identified by full command
  line carrying the authorized source digest before any signal was sent: learner PID
  2209 and `caffeinate` PID 2211. No other Phase 17 process existed.
- **SIGTERM only.** `SIGTERM` to 2209; it exited, and `caffeinate` 2211 exited with its
  child needing no separate signal. **SIGKILL was not required and was not used.**
- **No graceful handler exists.** There is no signal handling anywhere in the frozen
  closure, so `SIGTERM` ended the process immediately and `session.close()` did not run.
  This is exactly the case the operator instruction anticipated, and the terminal
  committed state is therefore the newest fully written telemetry row and the checkpoint
  it names.
- **An in-progress iteration was discarded: yes.** Iteration 536 was mid-step at signal
  time. It left **no artifact of any kind** — telemetry was byte-identical before and
  after (29,971,749 bytes, 535 lines), the checkpoint count did not change, no
  `joint_00536.pt` exists, and no temporary or partial file was left behind. All 535
  telemetry lines parse cleanly, including the last.
- **No trainer process remains.** No `run_phase17`, no `phase17`, no `caffeinate`, and
  no Python process of any kind belonging to this user.
- The source closure still digests to the authorized `4aa5c350…c349cc5` over 22 files,
  so nothing in the tree moved during the run.

### Terminal committed state

| | |
|---|---|
| Last fully committed iteration | **535** |
| Checkpoint | `checkpoints/phase17/RUN-2026-B/checkpoints/joint_00535.pt` |
| Generation | 535 |
| File sha256 | `362fd4a63f799795f2fc789bce0cbf3be0e8cd28c6462c0dc8ca256d758cdb4c` |
| Payload digest | `b6957238340b4bac66d43e2314983bc1a6a48646e8f28e1b8df3f4ac5dd3492f` |
| Both match telemetry | yes |
| Accepted under authorized identities | yes — `read_joint_checkpoint` accepted it with the run ID, config digest and source digest, refusing nothing |
| Highest checkpoint on disk | `joint_00535.pt` — no higher-numbered or temporary file |

**This checkpoint must not be evaluated.** It sits at iteration 535, past h12 at
iteration 510, and its EMA states (`6d52192f…`, `8f3b1f1d…`) were never exported as a
candidate. Agent 5 evaluates only the 25 frozen paired EMA candidates.

---

## 5. Late telemetry direction

Direction only, first ten iterations against last ten. No strength claim.

| Quantity | First 10 | Last 10 | Direction |
|---|---|---|---|
| Move entropy | 1.6706 | 0.3958 | down |
| Move mean KL | 0.01432 | 0.00309 | down |
| Move clip fraction | 0.1643 | 0.0375 | down |
| Move learning rate | 1.50e-04 | 1.84e-05 | down, by the frozen schedule |
| Setup empirical entropy | 1.6888 | 1.3737 | down |
| Setup final-epoch KL | 0.00686 | 0.00470 | down |
| Setup alpha | 0.0651 | 0.0152 | down, by the frozen anneal |
| Mean game length | 269.8 | 145.1 | down |

**No warning predicate fired at any point.** P6 came closest: the supervisor fixed its
first-hour median move entropy at 1.3389, putting the threshold at 0.3347. Move entropy
bottomed at 0.3777 and ended at 0.3981 — within about 13 % of the threshold, never
crossing it.

**Condition A6-C3 observed and recorded, not tuned.** The setup advantage's
entropy-to-outcome magnitude ratio ranged **0.491 to 4.596** over the run, median 1.073,
first-ten mean 1.727 against last-ten mean 1.067. It is volatile rather than trending —
an earlier reading of it as decaying with the alpha anneal did not survive the full
series. No compensating scale, floor, centering rule or controller was applied, as D10
section 4 requires.

---

## 6. Conditions carried from Agent 6

| Condition | Status |
|---|---|
| **A6-C1** 24-or-25 candidates | **Resolved: 25**, last ordinal h12 |
| **A6-C2** dry run omits two identities | **Discharged** — both recomputed before launch, h0 matched exactly |
| **A6-C3** D10-ADV-SCALE uncompensated | **Observed and recorded, not tuned** |
| **A6-C4** per-resume telemetry row loss | **No action** — closed by Agent 4C; this run never resumed |

---

## 7. What was not done

No evaluation of any candidate, checkpoint or raw weight set. No Agent 5 evaluator,
worker, publisher or transport process was started at any point during training. No
smoke test or test campaign was run. No change was made to the training source,
configuration or schedules. No candidate was fabricated, overwritten, renamed or
omitted. No resume was performed and none will be.

---

## 8. Handoff

`ready_for_post_training_evaluation: true`, on this basis: the trainer is stopped and no
Phase 17 process remains; all 25 candidate ordinals 0–24 are present, contiguous,
individually verified and byte-identical to write time; the terminal checkpoint verifies
against its telemetry-recorded file hash and payload digest and is accepted under the
authorized run, config and source identities; and the source closure still digests to
the authorized value.

Agent 5 evaluates **only** the 25 frozen paired EMA candidates in
`checkpoints/phase17/RUN-2026-B/exports`. It must not evaluate `joint_00535.pt`, any raw
move or setup weights, or any post-h12 state that was never exported.

### Notes Agent 5 may want

- Concentration appeared on **both** halves, the move half more sharply: move entropy
  1.67 → 0.40, setup empirical entropy 1.69 → 1.37.
- Mean game length fell 270 → 145 plies, which drove setup-episode arrivals up and is
  the sole reason iteration cost rose.
- Seat results are close to balanced across 233,936 games: 111,099 blue to 111,820 red,
  with a 4.7 % draw rate.
- The last 130 iterations of the frozen horizon were not run, so the move learning-rate
  and entropy schedules did not complete their anneal. The candidates are unaffected,
  since all 25 were published by iteration 510.

### Deliverables

```text
reports/phase17/phase17_run_closeout_v1.json
reports/phase17/agent_07_report.md
reports/phase17/agent_07_training_telemetry_summary.csv
reports/phase17/agent_07_candidate_ledger.json
```
