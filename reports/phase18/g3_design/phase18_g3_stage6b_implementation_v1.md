# Phase 18 Agent 6 — Stage 6B: the two-lineage pilot harness, built and tested

**Stage 6B of the G3 design work package: the implementation contract is frozen and the
harness is built and validated. No pilot was run, no full training run was started, no
evaluation game against a trained model was played, no held-out material was opened, and
nothing was pushed.** The only games this stage played are the tiny CPU smoke games inside
the test suite; they support no performance or timing claim.

Date: 2026-09-03. Author: Phase 18 Agent 6 (G3 design agent). Base: the approved Stage 6A
commit `7a37cde59f3d94dec3f2fbb66c47accc618f6001` on `phase18/g3-design`.

```text
branch                   phase18/g3-stage6b-harness (worktree output/phase18/worktrees/g3-stage6b/)
design source            reports/phase18/g3_design/phase18_g3_stage6a_joint_design_v2.md (7a37cde5)
reviewer decisions       fresh init both lineages; K = 64; canonical:live 1:1; live retention 32;
                         256 periods; bundle cadence 32; period-128 bundle SAVED, not evaluated;
                         bases 410-449 reserved; a second seed is never pooled to rescue; 6A
                         approval authorises no pilot and no full run (all implemented as such)
frozen contract          reports/phase18/g3_pilot/phase18_g3_pilot_contract_v1.json
test evidence            reports/phase18/g3_pilot/phase18_g3_stage6b_verification_v1.json
                         (+ g3_pilot_verification/junit_stage6b.xml)
runtime location         output/phase18/runtime/g3_pilot_v1/ (git-ignored, storage policy item 3)
protected file           reports/phase13/phase14_launch_manifest_v1.json: never staged, never modified
```

---

## 1. What was built (engineering items G3-ENG-01 … 05)

Everything lives in the additive Phase 18 namespace. No accepted Phase 2–17 module was
edited, and no existing Phase 18 module was edited either: the G2 raw-confirmation driver
pins the byte identity of the accepted setup implementation, so the buffer's exact state
capture lives in its own module (section 1.3).

```text
module                                          engineering item, and what it reuses unchanged
stratego/training/phase18/g3_contract.py        the frozen contract: PilotConfig (identical fields in both lineages
                                                except `lineage`; `setup_updates_enabled` is the ONLY switch and enters
                                                no seed), the frozen defaults, the case set, every stream seed through
                                                derive_stream_seed, the reserved-base refusal
g3_buffer_state.py                              exact capture / restore / digest of the accepted SetupBuffer's rows,
                                                counts, means, ready flags and counters, from outside the module
g3_live_store.py                                per-period live trajectory store (records + metadata + commit journal +
                                                finalisation summary); examples rebuilt by the accepted Phase 8
                                                builder `warmstart_examples.examples_for_game`
g3_collector.py          (G3-ENG-01)            the asynchronous pool-driven teacher-schedule collector: S slots, T plies
                                                per slot per period, cyclic cells, seeded pool pairing, two outcomes per
                                                finished game through `SetupBuffer.add_outcome`, G4 accounting; the game
                                                runner is `rule_population.play_corpus_game`'s decision loop split at
                                                the ply boundary; persistence through the Phase 17
                                                `capture_active_game` / `restore_active_game` codec verbatim
g3_c1.py                 (G3-ENG-02, C1 half)   `JointC1Trainer(WarmstartTrainer)`: the accepted trainer with ONE override
                                                (`_ensure_pipeline`) and one addition (`begin_period`); the mixed batch
                                                pipeline plans K batches per period from the frozen DataCursor, the
                                                retained live universe and a seeded draw; workers return arrays in plan
                                                order (the accepted `_BatchPipeline` argument)
g3_bundle.py             (G3-ENG-03)            the joint bundle: c1.pt (warmstart_checkpoint_v1, written by the accepted
                                                writer), setup/ (SetupTrainer.save_checkpoint), collector.pt (slot
                                                population + exact buffer state), manifest with lineage, switch,
                                                counters, every component sha256/state digest, live periods,
                                                bundle_id = sha256(manifest without the id); whole-bundle loads only
g3_pilot.py              (G3-ENG-02, loop)      LineageRunner: pool -> collect -> K C1 updates -> [candidate only:
                                                setup update + EMA] -> filter -> bundle; period records with every digest
                                                the gates read; `matching_check` (period-1 identity, equal budget,
                                                frozen control, moving candidate); resume from a bundle with live
                                                periods after the bundle discarded (renamed, never deleted)
g3_smoke.py              (design section 6)     the tiny CPU/one-thread/C0 smoke configuration and `restart_check`
                                                (uninterrupted vs save-at-n / new-process resume / period n+1)
g3_evaluation.py         (G3-ENG-04, 05)        the minimal G1 adapter: cases (160 bases x 8 opponents x 2 colours),
                                                own setups per case from the bundle's OWN setup model through the
                                                accepted `generate_pool`, the opponent's library base oriented to its
                                                colour, one MatchSpec per case under EVALUATION_RULES, the G1
                                                InferenceOwner + run_neural_schedule, retry-safe chunks, immutable
                                                receipts, planned = completed + failed + missing; the paired analysis
                                                (per-case EWR, per-base means, stratified cluster bootstrap within
                                                families with sqrt(n_f/(n_f-1)) rescaling, the LB > 0 AND point >= 0.05
                                                rule, the near-boundary flag)
scripts/phase18_g3_pilot.py                     the staged driver: --freeze, --verify, --launch-manifest, --run,
                                                --check-matching, --evaluate, --analyse, --restart-check
```

### 1.1 The period loop (both lineages, verbatim)

```text
(1) pool       raw setup actor samples 1,024 setups, snapshot_iteration = p - 1 in BOTH lineages (shared pool
               and reflection seeds); SetupBuffer.add_pool (S10 de-duplication, S23 counter reset)
(2) collect    every slot advances 202 plies in slot order; a finished game attributes its two outcomes
               (owner's perspective) and commits its trajectory + metadata to the live store; period accounting
               started = completed + in-flight delta + failed is checked before the period closes
(3) C1         64 supervised updates: each batch = 128 canonical keys (accepted cursor, accepted train order)
               + 128 live keys drawn without replacement from the last 32 finalised live periods by
               derive_stream_seed(namespace, 'c1_live_draw', seed, period, update); the accepted loss, clipping,
               AdamW, warm-up scheduler, counters and non-finite refusals run unchanged
(4) setup      candidate: SetupTrainer.update (five epochs over the ready rows) + one EMA update; a period with no
               ready row records an explicit skip. control: no update; the raw and EMA digests are asserted
               equal to the recorded initial version every period
(5) filter     rows older than 21 periods expire (S21: ceil(4000 / 202) + 1)
(6) bundle     every 32 periods and at the end; bundle_0 before any update
```

### 1.2 Decisions taken inside the design's envelope (recorded, not silent)

1. **Policy token.** Design item G3-ENG-04 asked for the bundle id "stamped into the policy
   token and every receipt" while gate G7 requires an identical schedule digest across arms.
   A per-arm token changes every match id and seed, so the arms share one token
   (`phase6_g3_bundle_greedy@0.2.0+float32`, the G1 construction) and the bundle id, lineage,
   period and component digests are stamped into every receipt and the arm record. G7 wins.
2. **Two retention quantities.** The reviewer's "retention = 32" is the live-example
   retention window; the setup buffer's storage duration stays 21 periods (S21, the absolute
   move limit in periods). Both are separate configuration fields and both are in the contract.
3. **Live decision sampler.** The accepted `warmstart_decision_sampler_v1` seeds its bins from
   a synthetic corpus game id through the Phase 8 master seed; a live game has no such id.
   The live store uses the same bins, widths and modulo draw (`decision_bin_bounds` imported
   unchanged) with bin seeds from the pilot namespace through `derive_stream_seed`.
4. **Live store format.** The accepted `CorpusWriter` refuses non-synthetic game ids by
   design (pre-commit metadata verification). The live store transcribes its frame layout,
   commit-after-flush order and journal-as-index rule; examples are built by the accepted
   builder. Nothing in the corpus path was edited.
5. **Short live stream.** When fewer than 128 live examples are retained (early periods, tiny
   smoke runs), the canonical half fills the batch so every update consumes exactly 256
   examples; the composition is recorded per update (`live_examples`, `canonical_examples`).
6. **Pool pairing.** The k-th game started in a period takes red row `k mod 512` and blue row
   `(perm_p[k mod 512] + k div 512) mod 512`, `perm_p` a seeded per-period permutation, so
   rows are reused about 3.9 times per period (design J2) with varied partners.
7. **Games terminal at creation.** A pool setup with no opening move loses at ply 0 under the
   engine's mobility rule (`phase2_1_reference_1.2.0`); the collector completes such a game
   at once with its real outcome and counts it (`immediately_terminal_games`). The G2 assay
   filtered these because its landscape had no notion of them; here the outcome is real.
8. **C1 validation cadence.** The accepted pilot-scope configuration validates every 500
   updates on the Phase 8 validation split (64 batches, evenly spread). This is retained
   unchanged (about every 7.8 periods); nothing is selected from it, and the time is recorded.
9. **Device split.** The setup model, its pools and its updates run on CPU (frozen:
   `setup_device = 'cpu'`), so the whole collector / outcome / setup-learning stream is
   bit-reproducible and identical across lineages by construction whatever the C1 device.
   C1 runs on the launch device (`mps` in the frozen contract). Consequently the period-1
   check requires exact C1 weight identity only on CPU and reports it on MPS (P18-D002).
10. **Single-process collection.** The collector advances its 2,560 slots sequentially in one
    process; C1 example building uses the accepted worker pool. Sharding slots across
    processes is possible without changing any identity (each worker owns fixed slots; the
    parent attributes in slot order) but was not built; see risk 5.

### 1.3 No existing module changed

A first draft added `capture_state` / `restore_state` methods to `setup_buffer.py`; the
G2 raw-confirmation driver's method-identity check (`tests/training/phase18/
test_g2_raw_confirmation_driver.py`) refused the changed digest, correctly. The capture now
lives in `g3_buffer_state.py`, which reads and rebuilds the buffer's rows and counters from
outside; `setup_buffer.py` and the package `__init__` are byte-identical to `7a37cde5`. The
round trip is proved exact (`test_g3_buffer_state.py`: identical digest, identical `process`
output and minibatches after restore).

---

## 2. The frozen contract (`phase18_g3_pilot_contract_v1.json`)

```text
run id / namespace       G3-PILOT-2026-A / phase18_g3_pilot_v1:7a37cde59f3d94dec3f2fbb66c47accc618f6001
                         (the namespace binds every seed to the approved design commit, the G2 lesson)
initialisation           C1 from the canonical Phase 8 seed 2026081302 (identity checked against the frozen
                         checksum cfe60bb0…e042b8 at run time); setup model from derive_stream_seed(namespace,
                         'model_init', 1) = 9033167075343681386; both lineages construct both from these seeds
frozen defaults          K = 64; 128:128 of batch 256; live retention 32; buffer storage 21; 256 periods; cadence
                         32; T = 202; S = 2,560; pool 1,024; TRAINING_RULES for collection; EVALUATION_RULES for
                         play evaluation (P18-A001); one seed; c1 device mps; setup device cpu
lineages                 candidate config digest and control config digest differ only in `lineage` /
                         `setup_updates_enabled`; matched_digest shared
evaluation               160 bases (400..409 x 16 families) x 8 handcrafted opponents x 2 colours = 2,560 cases per
                         arm; schedule digest frozen; opponents basic_heuristic, strategic_rule_based,
                         tactical_rule_based, stress_scout_rush, stress_miner_rush, stress_berserker,
                         stress_information_miser, stress_chaos; bases 410..449 refused by code; 10,000-replicate
                         stratified cluster bootstrap at 95%; PROCEED iff LB > 0 and point >= 0.05
arms                     candidate_final, control_final (primary); candidate_128, candidate_0, control_128 exist
                         only behind --diagnostic (reviewer decision 4)
```

The driver's `verify_frozen_identity` rebuilds both configurations and the 2,560-case schedule
from the code and refuses any drift before `--run`, `--evaluate` and `--analyse`.

---

## 3. Test evidence (the critical implementation checks)

Verification record `phase18_g3_stage6b_verification_v1.json` (driver `--verify`, from this
worktree before the commit): **77 passed, 0 failed, 0 skipped** over the eight targets, and
the restart check **PASS**. Full Phase 18 sweep (`tests/training/phase18` +
`tests/evaluation/phase18`, plus the Phase 17 checkpoint-codec tests and the Phase 8 trainer /
checkpoint tests the harness subclasses): **319 passed, 0 failed** (63.7 s), including the G2
raw-confirmation driver's method-identity tripwire (section 1.3).

```text
check (instruction)                          test / evidence                                                   result
identical before setup learning              test_g3_pilot::test_both_lineages_are_identical_before_setup_    PASS
                                             learning_begins: init digests (C1, setup, EMA, config, corpus)
                                             equal; bundle_0 components equal (only `lineage` differs);
                                             period-1 pool content, completed game ids, outcome records, live
                                             commit digest, C1 batch keys, live seeds AND C1 weight digest
                                             equal (CPU, one thread); the control's pools always come from
                                             the initial model; the candidate's diverge after its first update
control setup hash never changes             test_g3_pilot::test_the_control_setup_model_hash_never_changes:  PASS
                                             raw = EMA = init digest in every period record and every bundle
                                             (0..3); counters 0; verify_bundle refuses a control bundle that
                                             records an update or whose raw and EMA differ
candidate changes after a real update        test_g3_pilot::test_the_candidate_setup_model_changes_after_a_   PASS
                                             real_update: 5 epochs, >= 1 optimizer step, raw digest moved,
                                             EMA moved and != raw; a step that leaves the digest unchanged is
                                             fatal in the loop
own matched bundle only                      test_g3_pilot::test_each_lineage_uses_only_its_own_matched_      PASS
                                             bundle + test_g3_evaluation::test_cross_lineage_pairing_and_
                                             mismatched_digests_are_refused: cross-lineage verify/resume/
                                             evaluate refused (Phase18G3LineageError); a C1 and a setup model
                                             from different bundles or periods refused; a tampered component
                                             refused before any game
equal gameplay-update budget                 test_g3_pilot::test_both_lineages_receive_the_equal_gameplay_    PASS
                                             update_budget: K updates per period in both, identical global
                                             step, identical canonical cursor and live-draw seeds; matching_
                                             check compares totals
save-and-resume across an unfinished game    test_g3_pilot::test_checkpoint_save_and_resume_across_an_        PASS
                                             unfinished_game_reproduces_the_continuation (+ driver
                                             --restart-check): 3 games unfinished at the save point; a NEW
                                             PROCESS resumed bundle_1 and played period 2; identical completed
                                             game ids, outcome records, accounting, live commit digest, C1
                                             batch keys, C1 / setup raw / setup EMA / buffer digests, and
                                             bundle_2 component digests and counters
matched G1 evaluation schedule               test_g3_evaluation::test_the_schedule_is_identical_across_arms_  PASS
                                             with_matched_cases + test_a_tiny_schedule_plays_reconciles_and_
                                             pairs_across_arms: one schedule digest for both arms; per match
                                             identical unit id, seeds, colour, opponent, rules and the
                                             opponent's library formation; own setups differ only by model
finite losses, files written, duplicates     test_g3_pilot (all_finite, counters 0, every required file and    PASS
                                             manifest component present, tampered file / edited manifest
                                             refused); test_duplicate_boards_collapse_to_one_row_and_both_
                                             games_attribute_to_it (S10 collapse recorded; both games on the
                                             duplicated board attribute to the surviving row; no attribution
                                             failure)
supporting                                   collector: G4 accounting, exact capture/restore of unfinished     PASS
                                             games, unattributable outcome fatal, lineages identical in period
                                             1; live store: commit / finalise / verify / universe / examples
                                             identical to the accepted builder / never appended / discard;
                                             C1 mixture: plans pure in (cursor, universe, seed), exact resume,
                                             parallel loader == serial reference, unconsumed plans refused;
                                             driver: production config == frozen defaults, deterministic
                                             freeze, drift refused, diagnostic arms gated, horizon enforced
```

Also observed while testing (not gates): teacher games on fresh-model pool setups ran at about
10,000 plies per second in one process on this machine; random-vs-random cells lasted about
600 plies and heuristic cells 50–150. These are smoke observations that shaped the smoke cell
choice; they are not throughput claims for the pilot.

---

## 4. Remaining risks

1. **MPS non-reproducibility of C1 (P18-D002).** With `c1_device = mps` the period-1 check
   can prove identity of everything except the C1 weights; those are reported. If the review
   wants the C1 half of the check exact too, run both lineages with `--c1-device cpu` at
   `--freeze` time at the cost of wall time (unmeasured).
2. **Wall time and memory are unmeasured at scale**, by instruction. The design brackets
   4.1–10.8 wall hours for two lineages; the collector is single-process (risk 5), the live
   store writes about 25 MB per period per lineage (12.9 GB per pilot, design J3), and a
   bundle carries 2,560 captured in-flight games (compressed; the design's 46 MB estimate is
   for the components, not the population). Measured in the pilot.
3. **Validation exposure.** The accepted 500-update validation cadence reads the Phase 8
   validation corpus (formations from library bases 400–449). As in G1 nothing is selected
   from it; the reserved bases 410–449 are never opened as evaluation formations.
4. **Live-stream skew.** In early periods most games have not finished, so the live half is
   filled by the canonical stream; the composition is recorded per update and reported.
5. **Single-process collection.** If the pilot's collector throughput is far below the design
   bracket, sharding slots across processes is the remedy; it can be built without changing
   any seed or identity but was not built in this stage.
6. **Smoke evidence only.** Every end-to-end check ran on the C0 model, the six-game mini
   corpus and 3-slot / 64-ply periods. The production-scale paths (C1, the accepted corpus,
   2,560 slots, 1,024-setup pools, the 12-worker loader) are the accepted components run at
   their accepted scale, but the harness has not executed one production period. The pilot's
   first period is the first such execution and is bounded by the horizon and the bundle cadence.
7. **The G2 method-identity tripwire is load-bearing.** `test_g2_raw_confirmation_driver.py`
   refuses any byte change to the accepted setup modules; this stage tripped it once with a
   draft edit and moved the code out (section 1.3). Any later Stage 6C change to those
   modules will trip it again, by design.

---

## 5. The exact pilot command (proposed; NOT executed; requires an explicit written instruction)

From a clean detached execution worktree at the reviewed Stage 6B commit (storage policy:
under `output/phase18/worktrees/`), with the main repo's interpreter:

```bash
PY=/Users/brandonwashington/Dev/Github/stratego/gpt_agent/.venv/bin/python
COMMIT=<the reviewed Stage 6B commit>
cd /Users/brandonwashington/Dev/Github/stratego/gpt_agent
git worktree add --detach output/phase18/worktrees/g3_pilot_exec "$COMMIT"
cd output/phase18/worktrees/g3_pilot_exec
REPORTS=$PWD/reports/phase18/g3_pilot

# 1. bind the source, the contract and the (empty, git-ignored) runtime root; refuses a dirty tree
$PY scripts/phase18_g3_pilot.py --launch-manifest --source-commit "$COMMIT" --reports "$REPORTS"

# 2. the two lineages (separate processes; each verifies the accepted corpus and stops at period 256)
nohup caffeinate -i $PY scripts/phase18_g3_pilot.py --run --lineage candidate --reports "$REPORTS" \
    > /Users/brandonwashington/Dev/Github/stratego/gpt_agent/output/phase18/runtime/g3_pilot_v1_candidate.log 2>&1 &
nohup caffeinate -i $PY scripts/phase18_g3_pilot.py --run --lineage control --reports "$REPORTS" \
    > /Users/brandonwashington/Dev/Github/stratego/gpt_agent/output/phase18/runtime/g3_pilot_v1_control.log 2>&1 &
#    (a crash resumes from the last bundle with:  --run --lineage <l> --resume --reports "$REPORTS")

# 3. the period-1 lineage-identity and equal-budget check (may run as soon as both period 1 records exist)
$PY scripts/phase18_g3_pilot.py --check-matching --reports "$REPORTS"

# 4. the primary evaluation, control first (frozen order), on the G1 harness under EVALUATION_RULES
$PY scripts/phase18_g3_pilot.py --evaluate --arm control_final   --device mps --workers 8 --reports "$REPORTS"
$PY scripts/phase18_g3_pilot.py --evaluate --arm candidate_final --device mps --workers 8 --reports "$REPORTS"

# 5. the contrast, the ten gates and the decision input
$PY scripts/phase18_g3_pilot.py --analyse --reports "$REPORTS"
```

The launch manifest, matching record and results are written under `$REPORTS` and are
committed to the branch after review. Runtime data (bundles, live stores, receipts, period
records) stays under `output/phase18/runtime/g3_pilot_v1/` (git-ignored). Nothing continues
past period 256; the `--diagnostic` arms are not part of this command.

---

## 6. What this stage did not do; checks run

No pilot period, no production game, no evaluation game against a trained model, no held-out
access, nothing pushed; the approved commit `7a37cde5` and its history are untouched. Checks
run for this commit: the driver's `--freeze` (deterministic, re-derived by the driver tests)
and `--verify` (77 passed; restart check PASS) into `reports/phase18/g3_pilot/`; the Phase 18
sweep and the adjacent Phase 17 / Phase 8 suites; `git diff --check`. The protected Phase 14
manifest was never staged.
