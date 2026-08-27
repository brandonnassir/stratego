# Phase 15 — Agent 1
## Clean belief-corpus generation and B18/B24 fine-tuning

_Written 2026-08-24T13:45:55Z. This is an engineering deliverable: a corpus, two belief models and a handoff. It is **not** a playing-strength claim — no search was implemented, and none was evaluated._

## 1. Process boundary

This task does not authorize process control, and none was exercised: no emergency-stop file was created, no signal was sent, no live run state was edited, no checkpoint was rotated and no closeout command was invoked.

- verdict: `ready_for_compute` — 0 competing Phase 14 processes on a 14-core host
- Phase 14 run state at check time: 59.97 h elapsed, 102 iterations, `closed_reason='emergency stop'`, last written 2026-08-24T04:13:34Z
- the only Phase 14 process running was the read-only monitoring dashboard, which holds no compute
- Phase 14 artifacts were opened read-only: the candidate ledger, the two candidate evaluation weight files, and the two archive snapshots the ledger names

## 2. The orientation correction (section 4)

The old Phase 11B corpus and the Phase 12 match packs are contaminated: `Phase11BSetupSources.draw` returned canonical own-orientation tuples and the old glue handed them straight to `create_game()` for Blue. Canonical rank 0 is a player's own back rank, and Blue's engine setup order runs front-to-back, so an unoriented Blue army was placed reversed.

The rule Phase 15 enforces, re-derived from the engine's own `SETUP_SQUARES`:

```text
red engine row == canonical rank; blue engine row == 9 - canonical rank
```

The gate ran on **4,096 paired boards (8,192 armies)**, checking flag location, legal setup rows, complete inventory and Red/Blue paired orientation on every one.

- observed front-row flags: **145 of 8,192** (1.77%)
- the same draws under the old glue would have produced **3,154 of 4,096** Blue front-row flags (77.00%) — which reproduces Phase 12's 47-of-64 observation almost exactly
- negative canary: handing the gate a raw Blue canonical tuple is **detected** (`blue_canonical_passed_directly`)
- flag-row histograms are exact mirrors: Red concentrates on engine row 0, Blue on engine row 9

No Phase 15 module imports `belief/phase11b/corpus.py`, `Phase11BSetupSources` or `corpus_plans`, and a test enforces it.

## 3. P18 and P24 (section 3)

Both were resolved from the Phase 14 **candidate ledger**, not from the newest hot checkpoint, and every identity was re-derived from bytes.

**P18** — Phase 14 candidate hour 18

- archive snapshot `/Volumes/Brandon_Washington/stratego_phase14/archive/archive_0009.pt`
  - sha256 `c86d8c384daaa23c2ef8ad3d5196edabea60a97cce72b5e147453dd7f85473e6` (matches the ledger)
- model-state digest `9360f2add3dba31181bf41a59be66f3e0efe5b1e472c1cd2ebd17967a89b17e3`
- optimizer step 92,718, iteration 64, elapsed 18.15 h
- candidate evaluation complete: 128 games, mean EWR 0.8438, min stratum 0.7656 on pack `896a753b3d568902…`
- read-only Phase 15 copy `checkpoints/phase15/p18_source_readonly.pt` (sha256 `aa2cc39b3867264e…`, mode 0444)

**P24** — Phase 14 candidate hour 24

- archive snapshot `/Volumes/Brandon_Washington/stratego_phase14/archive/archive_0012.pt`
  - sha256 `3c393d0c8f8b233446febc8cca8ce7d1ceeee974231e6a9b53c7657a07d964f2` (matches the ledger)
- model-state digest `622d9e6caa723c932dedc5b77c257d532c1b0f8931f79851d863658f3cbbb87f`
- optimizer step 121,156, iteration 82, elapsed 24.18 h
- candidate evaluation complete: 128 games, mean EWR 0.7891, min stratum 0.6562 on pack `896a753b3d568902…`
- read-only Phase 15 copy `checkpoints/phase15/p24_source_readonly.pt` (sha256 `9bf256a9b085176b…`, mode 0444)

Neither file was written, and neither model was trained. The digests below are measured before and after each fine-tuning run.

- `p18` before `9360f2add3dba311…` → after `9360f2add3dba311…` — **unchanged: True**
- `p24` before `622d9e6caa723c93…` → after `622d9e6caa723c93…` — **unchanged: True**

## 4. `phase15_belief_corpus_v1` (sections 5–7)

`corpus_digest` `b0493a08d2fcb1dd6a2e234af0cde5bc3c5fe24dbc0bec0f86ab8de3c96dcbaa`

**155,027 eligible observer positions** carrying **4,373,492 supervised hidden pieces**, generated in 20.4 minutes on 8 CPU worker processes.

| split | positions | target | games | hidden pieces | pieces/position | library split |
|-------|-----------|--------|-------|---------------|-----------------|---------------|
| train | 120,010 | 120,000 | 7,558 | 3,385,258 | 28.21 | train |
| calibration | 15,004 | 15,000 | 948 | 424,676 | 28.30 | validation |
| development | 20,013 | 20,000 | 1,259 | 563,558 | 28.16 | validation |

Every split met its **initial engineering target**; the section 5 fallback floor was not used and no pilot evidence was needed to justify a reduction.

### Achieved mixture, counted over positions

Section 6 asks for counts *after position sampling*, not intended game counts. The training split's achieved shares:

- **observer**: p18 50.42%, p24 49.58%
- **opponent**: p18 24.00%, p24 25.16%, strategic_rule_based 10.48%, stress_information_miser 9.83%, stress_miner_rush 10.40%, stress_scout_rush 10.11%, tactical_rule_based 10.02%
- **setup source**: neutral_v1 34.81%, phase14_learned 45.43%, targeted_family 19.76%
- **observer colour**: blue 49.90%, red 50.10%
- largest absolute deviation from the design: observer_color 0.0010, observer_model 0.0042, opponent 0.0100, setup_source 0.0043
- targeted families missing: `none`
- game band: early 22.04%, late 52.00%, middle 25.96%

### Split disjointness

- `calibration|development`: 0 shared game ids, 0 shared public-state identities
- `train|calibration`: 0 shared game ids, 0 shared public-state identities
- `train|development`: 0 shared game ids, 0 shared public-state identities

Training draws its base setups from the accepted `train` library split. Calibration and development both draw from `validation`, which is not enough on its own — two games whose observer drew the same base setup reach the same opening public state. The `validation` population is therefore **partitioned in half, per family**, and each split draws only from its own half. This was found by the disjointness check on a first build of the corpus, which shared 43 opening positions between calibration and development; the corpus was regenerated after the fix.

### The public/privileged boundary

Two passes. The public pass plays the game with a policy that reads a `PolicyInput` and records only the ply, the unresolved-piece count and a sha256 of its own observation. The privileged replay rebuilds each selected decision from the action history, **checks the rebuilt observation against that digest**, and only then reads `dense_belief_target`.

Public arrays live in `public/`, the single label array in `privileged/`, and `load_split` returns the public half unless `labels=True` is passed.
- `calibration`: 424,676 stored ranks, all publicly admissible, all with remaining public inventory, 0 moved pieces carrying an immobile rank
- `development`: 563,558 stored ranks, all publicly admissible, all with remaining public inventory, 0 moved pieces carrying an immobile rank
- `train`: 3,385,258 stored ranks, all publicly admissible, all with remaining public inventory, 0 moved pieces carrying an immobile rank

Boards rebuilt from stored game ids alone: 512 armies re-checked, front-row flag rate 2.15%.

## 5. B18 and B24 (sections 8–10)

```text
frozen  P18/P24 prefix   first three C1 transformer blocks
trained copy             final C1 block + encoder norm
fresh   belief MLP       128 -> 512 -> 512 -> 12, GELU
```

A belief checkpoint contains only the copied block, the copied encoder norm, the belief MLP, the calibration temperature and the identity bindings — no policy tensor and no value tensor. Loading one requires the backbone whose model-state digest it recorded, and refuses any other.

### The recipe (declared once, shared by both)

```text
loss                       hidden_piece_cross_entropy
optimizer                  adamw
head_learning_rate         0.001
final_block_learning_rate  0.0001
weight_decay               0.0001
schedule                   cosine
batch_size                 256
max_epochs                 12
early_stop_patience        3
selection                  best_development_cross_entropy
```

- **B18**: 9 epochs run, best at epoch 6 (`patience`), 3.7 min total, 2.5 min to best
  - gradient isolation: 0 policy/value parameters carried a gradient, 66 checked
- **B24**: 6 epochs run, best at epoch 3 (`patience`), 2.5 min total, 1.3 min to best
  - gradient isolation: 0 policy/value parameters carried a gradient, 66 checked

### Calibration

- **B18**: fitted temperature 1.0431 on the calibration split (424,676 pieces), NLL 1.9932 → 1.9928
  - development NLL 1.9745 → 1.9743; ECE 0.0031 → 0.0057; top-1 unchanged: True
  - **kept: False**, applied temperature 1.0000
- **B24**: fitted temperature 1.0017 on the calibration split (424,676 pieces), NLL 1.9896 → 1.9896
  - development NLL 1.9709 → 1.9709; ECE 0.0048 → 0.0050; top-1 unchanged: True
  - **kept: False**, applied temperature 1.0000

## 6. Development metrics (section 11)

All models scored on the **same** 20,013 development positions and 563,558 hidden pieces, against the same accepted `remaining_count_belief_v1` denominator.

| model          | backbone   | CE     | R_CE   | R_CE 95% CI      | top-1  | Brier  | ECE    |
|----------------|------------|--------|--------|------------------|--------|--------|--------|
| B18            | P18        | 1.9745 | 0.9189 | [0.9157, 0.9219] | 0.2930 | 0.8086 | 0.0031 |
| B24            | P24        | 1.9709 | 0.9172 | [0.9141, 0.9202] | 0.2944 | 0.8077 | 0.0048 |
| Agent 1C       | Phase 9 C1 | 2.1479 | 0.9996 | [0.9957, 1.0030] | 0.2411 | 0.8468 | 0.0457 |
| count baseline | —          | 2.1488 | 1.0000 | —                | 0.2190 | 0.8612 | —      |

The uninformed floor — a flat 12-way vector — scores R_CE 1.1564, so every model above is better than knowing nothing.

### Paired comparisons (game bootstrap, same positions)

- `b18_vs_agent1c`: ΔCE -0.1734 [-0.1813, -0.1651] over 1,259 games — **B18 lower**
- `b18_vs_b24`: ΔCE +0.0037 [+0.0026, +0.0047] over 1,259 games — **B24 lower**
- `b24_vs_agent1c`: ΔCE -0.1770 [-0.1850, -0.1686] over 1,259 games — **B24 lower**

The Agent 1C comparison is on the **new** development corpus. Its old result (R_CE 0.9459) was measured on `phase11b_common_corpus_v1`, whose Blue setups are mis-oriented, and is quoted only to identify the artifact — never as the comparison set.

Scored here, Agent 1C reaches R_CE 0.9996 with a 95% interval of [0.9957, 1.0030] — that is, **statistically indistinguishable from the remaining-count baseline it was built to beat**. Two things changed at once and this experiment does not separate them:

1. the corpus is orientation-correct, and 1C was trained on boards where Blue's army was placed back-to-front;
2. 1C is attached to the accepted **Phase 9** backbone and was trained against a different observer, opponent and setup distribution than the one measured here.

So the drop should not be read as a clean measurement of the orientation defect's cost. What it does establish is narrower and sufficient for the handoff: on the corpus search will actually face, the surviving old belief model carries no usable advantage over the count baseline, and the two new specialists clearly do.

### Breakdowns

Full per-cell blocks — observer colour, observer source, opponent, opponent class, setup source, setup family and early/middle/late band — are in `agent_01_metrics.json` under each specialist's `development_calibrated.breakdowns`. The headline splits:

**B18**
- R_CE by observer color: blue 0.9205, red 0.9173
- R_CE by observer source: p18 0.9254, p24 0.9119
- R_CE by opponent class: neural 0.8972, rule 0.9356, stress 0.9409
- R_CE by game band: early 0.9636, late 0.8932, middle 0.9082

**B24**
- R_CE by observer color: blue 0.9186, red 0.9158
- R_CE by observer source: p18 0.9241, p24 0.9098
- R_CE by opponent class: neural 0.8951, rule 0.9353, stress 0.9391
- R_CE by game band: early 0.9631, late 0.8904, middle 0.9069

## 7. The belief/sampler interface (section 12)

```text
predict_marginals(public_state)      -> 12-way rank probabilities
sample_worlds(public_state, n, seed) -> complete legal hidden armies
```

Marginals go through the accepted constrained-world sampler by import: `stratego.evaluation.phase11_sampler.sample_belief_world`, unmodified. Pieces are not sampled independently and no accepted inventory or movement-impossibility constraint was altered.

- **B18**: 48 fresh positions, 384 sampled worlds — probabilities finite and summing to one (max deviation 4.44e-16); fixed seed reproduces worlds; remaining piece counts exact; moved pieces never assigned Flag or Bomb; every world passes the accepted validation stack
  - marginal latency: mean 2.09 ms, p50 2.08 ms, p95 2.31 ms
  - truth isolation: the public-state type carries exactly ['observation', 'public_state_document'], the provider reports `uses_hidden_truth=False`, and it answers from the public document alone
- **B24**: 48 fresh positions, 384 sampled worlds — probabilities finite and summing to one (max deviation 4.44e-16); fixed seed reproduces worlds; remaining piece counts exact; moved pieces never assigned Flag or Bomb; every world passes the accepted validation stack
  - marginal latency: mean 2.13 ms, p50 2.11 ms, p95 2.34 ms
  - truth isolation: the public-state type carries exactly ['observation', 'public_state_document'], the provider reports `uses_hidden_truth=False`, and it answers from the public document alone

## 8. The search handoff (section 13)

`reports/phase15/phase15_search_handoff_v1.json` binds exact digests for P18, P24, B18, B24, the corpus, both calibration values, the provider interface version and the accepted sampler version. It re-verifies against the bytes on disk: **verified = True** over 4 artifacts.

```text
P18  9360f2add3dba31181bf41a59be66f3e0efe5b1e472c1cd2ebd17967a89b17e3
P24  622d9e6caa723c932dedc5b77c257d532c1b0f8931f79851d863658f3cbbb87f
B18  b03685d3557d7c93ec5ce1c11b50907c13791a45e5a8359febe12b4f17612b99
B24  ac5e15b87f5c5cfd4e6fd5c6b004d56ac738f3994ec8b520bb32ef9e09b6cf4f
corpus b0493a08d2fcb1dd6a2e234af0cde5bc3c5fe24dbc0bec0f86ab8de3c96dcbaa
```

## 9. What this does and does not establish

- **Established**: an orientation-safe corpus of 155,027 positions; two belief specialists trained without touching P18 or P24; calibrated development metrics; providers that generate legal deterministic worlds; an exact handoff.
- **Not established**: any claim about playing strength. No search was implemented, no combined player was chosen, no Phase 12 artifact was modified and no Phase 14 task was controlled.
- The Phase 14 candidate EWRs quoted in section 3 are Phase 14's own 128-game pack results, reported to identify the checkpoints. They are not Phase 15 results.
- Corpus games were played with both neural seats in the accepted **greedy** decision mode; diversity comes from the setup mixture, not from sampled play. Each game runs to its accepted termination under `EVALUATION_RULES` (battleless 200, absolute 4000); the section 5 option to retire a trajectory early was not used, because evenly spaced sampling is defined over a game's complete eligible list.

## 10. Artifacts

```text
data/phase15/phase15_belief_corpus_v1/
data/phase15/phase15_belief_corpus_v1_manifest.json
checkpoints/phase15/p18_source_identity.json
checkpoints/phase15/p24_source_identity.json
checkpoints/phase15/b18_belief_v1.pt
checkpoints/phase15/b24_belief_v1.pt
reports/phase15/agent_01_process_boundary.json
reports/phase15/agent_01_orientation_gate.json
reports/phase15/agent_01_corpus_verification.json
reports/phase15/agent_01_learning_curves.json
reports/phase15/agent_01_metrics.json
reports/phase15/agent_01_interface_checks.json
reports/phase15/agent_01_summary.json
reports/phase15/agent_01_report.md
reports/phase15/phase15_search_handoff_v1.json
```

