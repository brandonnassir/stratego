# Phase 13 — Agent 1: Final Training Contract and Setup Census

Run date: 2026-08-20. Task: `instructions/phase_13_final_training_integration/01_AGENT_1_FINAL_TRAINING_CONTRACT_AND_SETUP_CENSUS.md`.

Every training-system decision required before the 168-hour Phase 14 run is now resolved and frozen in
`phase13_final_training_contract_v1.json`, and the one required setup-distribution census was performed
under a pre-declared alarm policy. **The census found no defect and no pathology on the production setup
path; `phase14_setup_source_v1` is frozen as the proposed 35/65 mixture without repair.** No RL training
was run. Phase 14 was not started. Agent 2 was not started.

## 1. What this task did and did not do

Did: read the accepted Phase 9 artifacts and froze exact continuation values; froze schedule, mixtures,
historical-pool algorithm, checkpoint hierarchy, wall-clock/deadline/storage semantics; wrote the census
alarm policy **before** sampling; ran the census; froze the setup source; built the fixed 128-game
candidate pack and the selection rule; added artifact tests.

Did not: train, sweep, tune, reopen Phase 10, touch the spent Phase 11 test bank, use search, or
reinterpret any closed phase. Phase 11 remains `phase11_final_classification = FAIL`,
`phase11_reinterpreted = false`. Phase 11B remains an engineering branch. Phase 12 search stays outside
Phase 14 training.

## 2. Starting model (contract section 2)

Phase 14 starts from the **accepted Phase 9 C1 checkpoint** — policy, value, and the accepted belief
auxiliary head:

```text
checkpoints/phase9/selfplay_c1_v1.pt
file sha256   dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea   (re-hashed this session)
model state   f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
```

Agent 1C is **not** used as the policy/value start; it stays reserved for the later search/belief
pipeline. (Note in passing: `checkpoints/phase12/phase9_c1_readonly_copy.pt` is a Phase 12 re-export
whose bytes differ from the original file; Phase 14 binds the original path and sha above.)

Resolved continuation posture: model weights from the frozen checkpoint; **fresh AdamW state**; KL
controller at the accepted initial beta 0.005; entropy coefficient constant at the accepted schedule's
terminal value 0.001. Rationale is recorded in the contract — Phase 14 is a new run identity, not a
resume of the Phase 9 run, and the accepted exact-resume semantics are defined only within a run.

## 3. Accepted Phase 9 configuration, retrieved (section 3)

All values read live from the frozen artifacts this session (none from memory). Verified live:
`contract_digest() == ad3dba3c4b7b...`, P9-C runtime identity `77af4d45dd8b...` reproduces, amendment
chain untouched.

| quantity | accepted value | source |
|---|---|---|
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 | `phase9_contract.OPTIMIZER_CONSTRAINTS` |
| LR9 | **3e-4** (P9-C) | frozen pilot matrix + runtime identity `77af4d45...` |
| LR schedule | constant (no warmup, no decay) | `OPTIMIZER_CONSTRAINTS` |
| policy objective | PPO clipped surrogate, eps 0.20, advantage-filtered decisions only | contract + `phase9_loss` |
| value objective | categorical CE vs soft W/D/L lambda targets, weight 0.5 (gamma 1.0, lambda_A 0.5, lambda_V 0.8) | contract |
| belief auxiliary | **present**, weight 0.25, accepted supervised targets (`phase9_example_v1` `a6b17a94...`) | contract + `phase9_loss` + `phase9_targets` |
| ratio clipping | 0.20; clip-fraction hard limit 0.75 | contract |
| behavior KL | D_KL(pi_b‖pi_theta), target 0.015, adaptive beta x2/x0.5 at 0.03/0.0075, clamp [1e-4, 0.2], hard veto 0.08; log floor 1e-12 | contract |
| advantage filter | \|A\| quantile 0.75, floor 0.01, retention 25%, standardization eps 1e-8; narrows the policy gradient only | contract |
| gradient clipping | global norm 1.0 | contract |
| EMA | **absent** from the accepted trainer — recorded as absent, Phase 14 runs without EMA | `phase9_trainer`/`phase9_checkpoint` inspection |
| batch/update | minibatch 512, 2 epochs/rollout, 2,048-game bulk-synchronous iterations, float32, fixed inference batch 64 | contract + config identity |
| entropy | accepted linear 0.005→0.001 over the run's own budget; Phase 14 resolution: constant 0.001 | contract; resolution documented |
| trajectory format | `trajectory_v1` + `phase9_example_v1` (`a6b17a94...`), float32 behavior probs in ascending absolute action order | frozen modules |
| game rules | `stratego_project_v1` / `board_10x10_v1`, first player red, battleless 100, absolute 4000, two-square off, chasing off | `warmstart_contract.CORPUS_RULES` |

Explicitly absent from the accepted artifacts (recorded, not guessed): LR warmup/decay, EMA, value-loss
clipping.

## 4–5. Continuation LR and schedule (sections 4–5)

Default proposal adopted — no accepted evidence supports any other pair:

```text
main continuation LR = 0.25  x 3e-4 = 7.5e-5     (constant within segment)
late continuation LR = 0.125 x 3e-4 = 3.75e-5    (constant within segment)
main segment 132 h  ->  late segment 36 h        (78.57% inside the 75–80% band)
transition_utc = run_start_utc + 475,200 s        (original wall-clock; downtime never moves it)
```

Transition semantics: the first collection unit launched at elapsed >= 132 h runs under late settings;
an in-flight unit finishes under main settings. LR sweeps, mid-run LR changes and LR raises are frozen
out.

## 6. Opponent mixture (section 6)

Exact scheduled counts per 2,048-game iteration (games are scheduled counts, never sampled — the
accepted Phase 9 idiom). Neural share 88.04% (band 85–90), handcrafted 11.96% (band 10–15):

```text
main:  current 1188 | historical 615 | handcrafted 245
late:  current  819 | historical 984 | handcrafted 245
handcrafted 245 = Strategic 61, Tactical 61, Scout-rush 41, Miner-rush 41, Information-miser 41
```

All five handcrafted families exist in the accepted rosters (`strategic_rule_based`,
`tactical_rule_based`, `stress_scout_rush`, `stress_miner_rush`, `stress_information_miser`).
Colour-balance and learner-control semantics stay bound to the accepted Phase 9 contract. No mixture
sweep.

## 7. Historical archive vs active pool (section 7)

Archive: durable snapshot every 2 h of original elapsed wall-clock, no tournament, everything kept.
Active pool `phase14_active_pool_v1`: 16 members = anchors **P8** (`warmstart_c1_v1.pt`,
`f7e9c40d...`) and **P9** (`selfplay_c1_v1.pt`, `dfd698e5...`) + up to 14 snapshots; weights
anchors 20 / older 25 / middle 25 / recent 30 with equal split within a category and proportional
redistribution from empty categories.

Membership is a pure function f(k) of the ordered archive: k <= 14 keeps everything (contiguous
age bands); k > 14 takes the 6 newest, then 4 quantile picks from each half of the remainder. The
historical bucket (615/984 games) is partitioned into exact per-member counts by largest remainder
with rotating tie-break. Pool, categories and cursors live in every hot checkpoint; resume recomputes
f(k) and refuses on any mismatch. The reference implementation in
`tests/training/test_phase13_agent01.py` sweeps k = 0..89 (a 168-hour run archives 84) and verifies
well-formedness, so Agent 2 implements from an already-executable spec.

## 8–10. Belief auxiliary, search prohibition, checkpoints (sections 8–10)

- Belief auxiliary: present in the accepted learner; **retained unchanged** (weight 0.25, same
  targets). No defect prevents its use.
- Search: TINY / SMALL / MEDIUM all `NOT USED`; no search in action selection, targets, opponents,
  improvement or trajectory generation. Phase 14 trains the direct C1 system only.
- Checkpoints: hot every 15 min (>= 4 retained, fast internal storage, full resume state list frozen —
  the EMA slot records "absent"); durable archive every 2 h on the external volume; final-policy
  candidates every 6 h = hours 0, 6, ..., 162, 168 (29 candidates, a marked subset of the archive).
  Hour 168 is a candidate, not automatically the deployed policy.

## 11–12. Candidate pack and selection rule (sections 11–12)

`phase14_checkpoint_selection_pack_v1.json` — 128 games per candidate: 32 each vs accepted Phase 9
(greedy), Strategic, Tactical, Scout-rush; 16 red / 16 blue per stratum; both boards of every game
drawn from the frozen production source at pack ordinals 1,000,000+ (disjoint from census ordinals);
opponent/candidate decision seeds frozen per game; EVALUATION_RULES; every resolved board carries its
oriented engine setups and passes engine validation (4 of 256 sides carry a front-row Flag — the
by-design census rate, unlike the mis-oriented Phase 12 pack's 47 of 64 boards).
`pack_content_digest 896a753b3d568902e93e803f1a45de9e8834ff1cdf90bc08cfacf90bcf0c2bde`, bit-identical
across two independent builds.

Selection rule (`phase14_checkpoint_selection_rule_v1.json`): highest equal-weight mean EWR over the
four strata; tie-break 1 highest minimum-stratum EWR; tie-break 2 later checkpoint. No confidence
intervals, no reweighting. Evaluations are isolation-only monitoring: they can never stop training,
change LR/mixture/source/pool/cadence, or extend the deadline; a failed evaluation is rerun later on
the same frozen pack.

## 13–14. Wall-clock contract and deadline behavior (sections 13–14)

`run_start_utc` stamped immediately before the training loop; `run_deadline_utc = run_start_utc +
604,800 s`; both persisted in every hot checkpoint; `remaining_time = run_deadline_utc − now` on every
restart; a fresh deadline is never created; downtime counts. At/after the deadline: stop launching
collection units → finish/discard the active bulk unit at the accepted bulk-sync boundary → no new
optimizer step → final run-state checkpoint → preserve hour-168 candidate → final counters + manifest →
mark closed. Post-deadline recovery refuses optimizer steps and finalizes.

## 15. Setup census (section 15) — the measurement

**Ordering.** `phase13_setup_census_alarm_policy_v1.json` was written at 2026-08-20T20:51:35Z, before
any sampling; the census runner structurally refuses to run without it and embeds its sha256
(`12769b84...`) in the census artifact. Thresholds were chosen from prior accepted evidence only and
were not moved.

**Method.** Train split, accepted entry points only, frozen selector-audit seed stream (ordinals
0..8191 per color): 16,384 production-mixture draws, 16,384 standalone `neutral_v1` draws under the
same seeds, 10,666 learned-branch draws (>= 10,000 required), 1,024 neutral-branch adapter equality
checks, and 8,192 paired boards built through the production orientation path into real engine states
with red's ply-1 legality enumerated by the engine itself. Full library enumeration (8,000 bases) for
attribution. Runtime 16.5 s; the whole census artifact reproduces bit-identically on a second run
(`census_content_digest a7103107b7c77a5b...`).

**Results.**

```text
defects:            0 of every class (D1 flag-movement, D2 family-range, D3 invalid placement,
                    D4 coordinate transform, D5 reflection, D6 adapter equality)
P_trivial:          0.003296   (27/8192 boards; predeclared pathology line 0.05)
P_predecision:      0.001953   (16/8192 boards, engine-verified; line 0.025)
classification:     VALID_BUT_STRATEGICALLY_POOR (legal, by-design exposure only)
```

Flag row distribution (final draws): production mixture front-row rate 1.43% red / 1.20% blue;
neutral_v1 1.65–1.83%; learned branch 1.00–1.16%. Sampled rates match the **exact** distribution
expectations computed from the frozen selector vectors (neutral 1.83%, learned 0.89–1.09%, mixture
1.22–1.35%). Library composition: families F00–F14 place every Flag in the back two ranks exactly as
contracted; **all front-row Flags in the library and all 215 front-row draws in the census belong to
F15 `irregular_high_entropy`** (117/400 train bases front-row), whose contract admits flag_rank 0..3.

Attribution, as required by the instruction:

- **neutral generation**: sole structural origin (F15 bases), at the contracted rate;
- **learned selection**: *reduces* exposure — P10-D puts 4.5–5.2% mass on F15 vs the neutral 6.25%;
- **reflection**: changed the Flag row 0 times in 32,768 draws (file mirror only; the open-file set is
  mirror-symmetric, so lane exposure is invariant too);
- **perturbation**: changed the Flag row 0 times (Flag cell pinned by the frozen contract);
- **paired interaction**: immediate lanes need a front-row Flag *and* an opposing front-row Scout in
  the same open file — 27/8192 boards, spread across branch pairs in proportion (9 learned+learned /
  8 / 7 / 3), no amplification. Engine legality agreed with the geometric model on all 8,192 boards.

The rates sit exactly in the accepted Phase 9 regime (~0.24% observed 1–2-ply endings), so the
production distribution Phase 14 will train on is the one the accepted run already trained on.

**Why Phase 12 saw 47/64 boards with front-row Flags.** Pre-census code reading located the cause
outside the production source: `stratego/belief/phase11b/corpus.py` `Phase11BSetupSources.draw()`
returns the **canonical (own-orientation) tuple**, and Phase 11B corpus construction and Phase 12
matchplay pass it directly to `create_game()` without `oriented(player)`. Red's map is the identity so
red is unaffected, but **every blue army in Phase 11B corpus games and Phase 12 match games was placed
back-to-front** — blue's back-rank Flags (~79% of bases) landed on blue's front row. The census
confirms the production path (`LibrarySetupSource.assign` / `SelectorDraw.oriented`) is clean: D4 = 0
over 8,192 engine boards. Consequences for Phase 11B/12 conclusions are out of this task's scope (a
repair task for that glue already exists from Phase 12); Phase 14 never touches that code path, and
both `phase14_setup_source_v1` and the contract carry an explicit integration warning naming it.

## 16. Setup repair rule (section 16)

No defect, no pathology → no repair, no new implementation identity. **Frozen:**

```text
phase14_setup_source_v1 = 35% neutral_v1 + 65% accepted P10-D learned selector
                          accepted reflection/perturbation behaviour, train split,
                          orientation via the accepted oriented(player) path
```

P10-D bindings verified this session: selector config sha `6e227815bc3cb44f...` (re-hashed), utility
file `50cb947dae63...` (re-hashed), model_T coefficients `d898782a...`, library file `cd8c3921...`,
library content digest `7b8a6660...`, production train distribution digests red `9ac5b52e...` / blue
`abef2299...`. Phase 10 evidence untouched.

## 17. Storage / retention (section 17)

Volume inspected 2026-08-20: `/Volumes/Brandon_Washington`, 1000 GB, **994 GB available**. Projection:
planning rate 3.572 GiB/h → ~600 GiB raw over 168 h (the Phase-9-loop-measured basis, 27,096.7 B/game
at ~1.86 games/s, gives only ~28 GiB, so 600 GiB is the conservative ceiling); + ~4 GiB archive
snapshots + ~1 GiB hot/logs/evals; ×1.2 reserve ≈ **726 GiB < 994 GiB → full raw retention fits and is
planned.** A contingency rolling policy is pre-authorized only if free space drops below 120 GiB:
delete already-consumed Phase 14 raw shards oldest-first, keep all checkpoints/metrics/snapshots plus a
1-in-16 representative shard sample. Earlier accepted evidence is never deleted under any condition.

## 18. Monitoring without tuning (section 18)

The 23-item metric list is frozen in the contract. The control surface must not offer live edits of
LR, loss weights, opponent mixture, setup source, pool algorithm, selection rule, or deadline;
emergency stop remains available.

## 19. Deliverables

```text
reports/phase13/phase13_final_training_contract_v1.json
reports/phase13/phase13_setup_census_alarm_policy_v1.json     (written before sampling)
reports/phase13/phase13_setup_census_v1.json                  (bit-reproducible, digest a7103107...)
reports/phase13/phase14_setup_source_v1.json                  (FROZEN)
reports/phase13/phase14_checkpoint_selection_pack_v1.json     (bit-reproducible, digest 896a753b...)
reports/phase13/phase14_checkpoint_selection_rule_v1.json
reports/phase13/phase13_agent_01_report.md
reports/phase13/phase13_agent_01_summary.json
scripts/run_phase13_agent01.py                                (census + pack stages, policy-gated)
tests/training/test_phase13_agent01.py                        (15 artifact tests, all passing)
```

Full suite after this task: **6,162 passed / 3 skipped** in 352 s (was 6,147 / 3 after Phase 12 Agent 5;
the 15 new tests are the Phase 13 artifact tests). No accepted file was modified — `git status` shows
only new untracked files.

## 20. Stop condition

All section-20 items are frozen: Phase 9 values retrieved and bound; continuation LRs; main/late
schedule; opponent mixtures; archive/active-pool rules; belief auxiliary treatment; checkpoint
hierarchy; candidate pack and selection rule; alarm policy written before the census; census completed;
no repair required; final setup source frozen; storage/deadline/recovery semantics frozen. **No RL
training was run. Agent 2 does not begin automatically.**
