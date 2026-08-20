# Phase 11B Closure Report

- **Agent:** Phase 11B Closure Agent (`instructions/phase_12_sequential_agent_plan/01_PHASE_11B_CLOSURE_AGENT.md`)
- **Date:** 2026-08-20 (UTC)
- **Git base:** `3a0720c6745bfa3688e0cd185813a1004e8bd8eb`
- **Formal status carried unchanged:** `phase11_final_classification = FAIL`, `phase11_reinterpreted = false`, `phase11_test_bank_spent = true`, `phase12_was_not_authorized_by_phase11 = true`

Phase 11B is closed. This report records the closure decisions; the machine-readable
artifacts are `phase11b_engineering_selection_v1.json` and `phase12_handoff.json`
in this directory. No committed Phase 9/10/11/11B artifact was modified.

## 1. Agent 4 — finished and preserved

The Agent 4 hybrid raw+C1 experiment had already completed (2026-08-20T03:53:25Z)
before this closure ran; no training process was active. Nothing was re-run,
lengthened, or re-evaluated. Its full record — checkpoint, architecture/config,
parameter count, learning curve, training wall-clock, time-to-best, development
metrics, standardized summary, and report — is committed at `3a0720c`:

- `agent04_hybrid_raw_c1_cnn`: **R_CE 0.9614** [0.9582, 0.9646], top-1 0.2561,
  3,897,724 parameters, 105.8 s training, best at 14.9 s (step 91, 0.86 epochs).
- Checkpoint `checkpoints/phase11b/agent04_hybrid_raw_c1_cnn.pt`
  (`e25afd24…`) matches its recorded digest on disk today.
- Finding preserved as recorded: **raw and C1 are not complementary.** Hybrid −
  C1-only = −0.0010 R_CE, paired ΔCE CI [−0.0064, +0.0021] straddles zero;
  adding C1 to raw helps a lot, adding raw to C1 adds nothing measurable.
- Repeat run under identical seed/config was bit-identical. Suite at its close:
  6,016 passed / 3 skipped.

Agent 4's row remains on the leaderboard below for future reference; per the
closure instruction it could not change the selection.

## 2. Agent 5 — cancelled

```text
phase11b_agent5_status = cancelled_by_instructor
```

The autoregressive-Transformer candidate was not implemented, trained,
benchmarked, or partially initialized. Its specification remains preserved at
`instructions/phase_11b_belief_engineering_sprint/05_AGENT_5_AUTOREGRESSIVE_TRANSFORMER.md`.

## 3. Selection — Agent 1C

```text
selected_candidate       = Agent1C (agent01_1c_final_block_plus_mlp)
selected_candidate_R_CE  = 0.9460
selected_candidate_top1  = 0.2640
selection_type           = engineering
scientific_claim         = none
```

Agent 1C is *a copy of the accepted Phase 9 C1 in which the larger belief MLP
and final C1 block were fine-tuned for supervised belief prediction*. It reads
the frozen C1 penultimate token field `[B, 100, 128]` and re-runs a re-trained
copy of the final block (198,272 params) plus final encoder LayerNorm (256)
into a new 334,860-parameter MLP head (128→512→512→12, GELU) — 533,388 trained
parameters total. The accepted Phase 9 checkpoint itself was never modified.

Selection authority: the closure instruction (§3) selects 1C, the leader by
R_CE, regardless of Agent 4's result. Agent 1's own report had preferred 1B
under the Engineering Winner Rule (inside the 0.005 band, cheaper, no accepted
C1 weight retrained); both readings are recorded and Agent 1's report is
unmodified. 1B remains the natural fallback if 1C's unfrozen block proves
awkward in search integration.

### Checkpoint provenance finding (recorded, not repaired)

Digest verification during closure found that Agent 1's `--repeat-train`
reproducibility pass **overwrote the first-pass checkpoint files in place** for
1B and 1C (1A's two passes were bit-identical). Evidence: the surviving files'
embedded and independently re-scored dev metrics equal Agent 1's recorded
`r_ce_repeat_pass` values exactly (1B to all 16 digits), the mtimes fall inside
Agent 1's run window, and the recorded first-pass SHAs match no file on disk.
The drift between passes is 3.2e-5 (1C) / 6.4e-5 (1B) R_CE — Agent 1's own
measured run-to-run training noise; no ranking or selection changes under
either reading.

**Phase 12 therefore binds to the surviving bytes:**

```text
path               checkpoints/phase11b/agent01_1c_final_block_plus_mlp.pt
sha256             a125208605f5e68c897214016e1803718439755e6286e5a185447636ffcd9fad
state_dict_digest  69104cd98c66ae1715b93990cc949b50ebad47bf66177b0d38eb6c958db7c2b8
dev R_CE           0.9459282341313351   (headline 0.9460 holds for both passes)
dev top-1          0.2638727548923242   (0.2640 first pass / 0.2639 surviving)
```

Training corpus: `phase11b_common_corpus_v1` train split (2,048 games, 26,898
samples, 817,255 hidden pieces, setup-library split `train`); development
corpus: its dev split (512 games, 1,828 samples, 55,955 pieces, library split
`validation`); corpus digest `903bf10a…`, all 12 data files re-verified today.

## 4. Final Phase 11B leaderboard

Dev R_CE against `remaining_count_belief_v1` (CE 2.1949); uniform floor 1.1321.

| # | Candidate | R_CE | top-1 | Params | Note |
|---|-----------|------|-------|--------|------|
| 1 | `agent01_1c_final_block_plus_mlp` | **0.9460** | 0.2640 | 533,388 | **selected**; surviving file is the repeat pass (0.94593) |
| 2 | `agent01_1b_attached_mlp_head` | 0.9495 | 0.2603 | 334,860 | Agent 1's in-band economy pick; fallback |
| 3 | `agent01_1a_existing_linear_head` | 0.9531 | 0.2542 | 1,548 | same head as Phase 11, retrained |
| 4 | `agent04_hybrid_raw_c1_cnn` | 0.9614 | 0.2561 | 3,897,724 | not complementary with C1 |
| 5 | `agent03_c1_feature_cnn` | 0.9624 | 0.2569 | 3,898,444 | C1 features beat raw pixels |
| 6 | `agent02_raw_observation_cnn` | 0.9686 | 0.2520 | 3,897,004 | corpus-bound, overfits from epoch 2 |
| 7 | `phase11_head_unchanged_reference` | 0.9834 | 0.2303 | 1,548 | accepted Phase 11 head, unchanged |

Interpretation (as required by the closure contract): dedicated supervised
belief training produced the dominant gain; enlarging the head helped only
modestly; allowing the final C1 block to adapt produced the best result;
raw-observation and C1-feature CNNs were worse despite being larger; C1's
representation is highly useful and sample-efficient at the current data scale.

## 5. Preservation verification

Re-verified from live bytes today, all matching their committed records:
15/15 preserved Phase 11 artifact digests (contracts, banks, ledger, system
record, report, and the seven frozen source modules), 12/12 common-corpus file
digests plus the manifest digest, the accepted Phase 9 checkpoint SHA
(`dfd698e5…`), and the 1A/A2/A3/A4 candidate checkpoints. The only
discrepancy is the 1B/1C repeat-pass overwrite documented above, which predates
this closure and is now permanently recorded in the selection artifact.

## 6. Phase 12 handoff

Written to `reports/phase11b/phase12_handoff.json`. Roles:

```text
Phase 9 C1  ->  policy/value    (checkpoints/phase9/selfplay_c1_v1.pt, sha dfd698e5…)
Agent 1C    ->  belief only     (sha a1252086…, state digest 69104cd9…)
```

Do not assume Agent 1C's policy/value outputs remain production-compatible
after its final block was fine-tuned. The handoff also fixes the identities of
the original Phase 11 belief (`phase11_system_v1`, system digest `e4452ba3…`,
head `a9df48a1…`), the `remaining_count_belief_v1` baseline
(`phase11_baselines.py`, `5f84c459…`), the accepted sampler interface
(`phase11_sampler.py`, `a0119f01…`, used unmodified via
`phase11b_belief_interface_v1`), and this closure's selection artifact
(`af2d293d…`). The oracle belief provider is diagnostic-only and never
available in production.

## 7. Stop

Per the closure contract, this agent stops here: Agent 4's result preserved,
leaderboard final, Agent 5 cancelled, `phase11b_engineering_selection_v1`
created, Phase 12 handoff written. **No search implementation was begun.**
Phase 12 Agent 1 (Search Core) is the next step after review.
