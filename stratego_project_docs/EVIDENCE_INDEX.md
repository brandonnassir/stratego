# Evidence and artifact-status index

**Written 2026-08-27. Updated 2026-08-31 by Phase 18 Agent 1** (§7 added; §6.1 amended); **updated 2026-09-02 by Phase 18 Agent 4** (§7.2b: G1 accepted, review and errata rows).
A compact classification of every major artifact.
This index deliberately does **not** repeat all metrics — it links. Values live
in the reports; the reports were not edited.

Labels: `ACCEPTED` · `ENGINEERING` · `INCOMPLETE` · `INTERRUPTED` ·
`SUPERSEDED` · `CONTAMINATED` · `PENDING` · `HISTORICAL`. Definitions in the
root [`README.md`](../README.md) §7.

---

## 1. Models and checkpoints

| Artifact | Identity | Status | Notes |
|---|---|---|---|
| `checkpoints/phase9/selfplay_c1_v1.pt` | `dfd698e5b6cf536a…` | **`ACCEPTED`** | **The latest accepted direct policy.** Sealed vs-anchor EWR 0.8442. [Phase 9 report](../reports/phase_9_implementation_report.md) |
| `checkpoints/phase12/phase9_c1_readonly_copy.pt` | `81906f71…` | `HISTORICAL` | Convenience re-export with **different bytes**. Never the binding target. Git-ignored. |
| Phase 8 anchor checkpoint | see [Phase 8 report](../reports/phase_8_implementation_report.md) | `ACCEPTED` | Used as an **evaluation opponent** thereafter, not as a playing model. |
| Phase 10B fine-tuned checkpoints | `checkpoints/phase10b/` | **`INCOMPLETE`** | Run paused at 5 of 30 iterations. No classification, no selection. Git-ignored. |
| Phase 11B candidates | `checkpoints/phase11b/` | **`CONTAMINATED`** | Trained on the mis-oriented corpus. Agent 1C is the selected one (`a1252086…` surviving sha). Repeat-training overwrote the 1B/1C first-pass bytes — bind to surviving bytes. Git-ignored. |
| **P18** — Phase 14 hour-18 candidate | model state `9360f2add3dba311…`; snapshot `archive_0009.pt` (`c86d8c384daaa23c…`) | **`INTERRUPTED` (intermediate)** | Not accepted, not selected. In-repo read-only copy `checkpoints/phase15/p18_source_readonly.pt`. |
| **P24** — Phase 14 hour-24 candidate | model state `622d9e6caa723c93…`; snapshot `archive_0012.pt` (`3c393d0c8f8b2334…`) | **`INTERRUPTED` (intermediate)** | Same. The backbone Phases 15–16 actually built on. |
| Phase 14 candidates hours 0/6/12/30/36/42/48/54 | `…/stratego_phase14/evaluations/weights/hour_0NN.pt` | `INTERRUPTED` (intermediate) | Exist and are evaluated on the frozen 128-game pack; hours 60–168 do **not** exist. |
| **B18** belief specialist | `b03685d3557d7c93…` | **`ENGINEERING`** | Belief-only; requires the P18 backbone by digest. `checkpoints/phase15/b18_belief_v1.pt` |
| **B24** belief specialist | `ac5e15b87f5c5cfd…` | **`ENGINEERING`** | Belief-only; requires the P24 backbone by digest. `checkpoints/phase15/b24_belief_v1.pt` |
| Phase 16 training arms | `checkpoints/phase16/arms/` | **`ENGINEERING`** | Three 6-hour arms from P24. No arm moved measurably; not a deployment candidate. |

---

## 2. Playing candidates (search configurations)

| Artifact | Status | Why |
|---|---|---|
| `checkpoints/phase12/phase12_search_candidate_v1.json` | **`SUPERSEDED`** + evidence **`CONTAMINATED`** | Phase 9 C1 + Agent 1C at TINY. Its EWRs (0.6406 TINY / 0.6875 MEDIUM) come from mis-oriented packs. The *engine* it names (`phase12_root_world_search_v1`) is still the accepted engine everything runs on. |
| `checkpoints/phase15/phase15_search_candidate_v1.json` | **`SUPERSEDED`** by Phase 16 | P24 + B24, TINY default / MEDIUM max. `scientific_validation_status: not performed`. Its +0.1375 search advantage **did not reproduce** on `phase16_benchmark_v1`. |
| `checkpoints/phase16/phase16_stochastic_candidate_v1.json` | **`ENGINEERING` — current candidate** | `stoch_t015_r100` over P24+B24. **Selected for unpredictability, not demonstrated strength** (+0.0167 ± 0.0640 at MEDIUM on its own pack). Not promoted to production — the production agent never ran. |
| `checkpoints/phase16/phase16_recipe_candidate_v1.json` | **`ENGINEERING` — no recipe adopted** | Records `adopt_recipe.pass = False` and the `STOP` verdict. **Explicitly authorizes no long run.** |

---

## 3. Corpora, packs and libraries

| Artifact | Status | Notes |
|---|---|---|
| `data/setups/setup_library_v1.jsonl` + manifest | **`ACCEPTED`** | 8,000 boards, 16 families × 500; 6,400 / 800 / 800 train / validation / test. Content digest `7b8a666…`. |
| `evaluation_setup_bank_v1` (Phase 4) | `ACCEPTED` | 1,024 fixed pairs. **Not** the training setup generator. |
| `phase10_setup_selector_v1` (`5e2b9c3a…`) | **`ACCEPTED`** | Phase 10, `PASS-NONINFERIOR`, 8/8 gates. Non-inferior, **not** superior. |
| `phase11_test_bank_v1` | **`HISTORICAL` — permanently spent** | Opened exactly once, by Phase 11 Agent 7. A future belief-repair phase needs fresh sealed evidence. |
| `phase11b_common_corpus_v1` (`903bf10a…`) | **`CONTAMINATED`** | Blue setup-orientation defect. Invalid for belief-quality or strength claims. Kept as record. |
| Phase 12 match packs | **`CONTAMINATED`** | Same defect; 47 of 64 boards observed with front-row flags. |
| `phase14_checkpoint_selection_pack_v1` (`896a753b3d568902…`) | `ACCEPTED` (as an instrument) | 128 games / 4 strata. **Noisy, not inflated** — mean delta +0.0066 vs a 2,200-game re-read. |
| `phase15_belief_corpus_v1` (`b0493a08d2fcb1dd…`) | **`ENGINEERING` — orientation-correct** | 155,027 positions / 4,373,492 hidden pieces; splits provably disjoint. Supersedes `phase11b_common_corpus_v1`. |
| `phase15_match_pack_v1` (`f2e2e7a4504ea271…`) | `ENGINEERING` | 120 paired boards. Fresh, orientation-gated. Valid on its own terms; **its search-advantage reading did not reproduce**. |
| `phase16_benchmark_v1` (`ebd13019…`) | **`ENGINEERING` — canonical instrument** | 120 paired boards + predeclared `quick60`. Executable manifest: rebuilds every board from its id and refuses on any byte difference. |
| `phase16_adversarial_setups_v1` (`e01529ce…`) | `ENGINEERING` | 9 families / 96 setups; authored digest `dcafa161…` frozen. `operator_harvest` present but **empty**. |
| `phase16_adversarial_baseline_v1` (`e937df0d…`) | `ENGINEERING` | 96 paired board triples, 3 arms, at TINY and MEDIUM. |
| `phase16_agent02_interim_pack_v1` | `ENGINEERING` | 60 paired boards; the pack the stochastic selection was decided on. Interim fallback source. |

**Rule for all of the above: an EWR belongs to its pack.** Phase 16 measured a
sign flip between two packs drawn from the *same* library split by the *same*
machinery. Cross-pack comparison is forbidden in conclusions.

---

## 4. Run state and evaluation records

| Artifact | Status | Notes |
|---|---|---|
| `/Volumes/Brandon_Washington/stratego_phase14/phase14_run_state.json` | **`INTERRUPTED` — read-only** | `closed: false`, `closed_reason: "emergency stop"`, 59.97 h / step 202,504 / iteration 102. **Do not write to it.** |
| `…/stratego_phase14/phase14_emergency_stop.json` | `HISTORICAL` | The operator's stop record and its stated reason. |
| `…/stratego_phase14/evaluations/phase14_candidate_ledger.json` | **`ACCEPTED` (as a record)** | The authoritative source for P18/P24 identity and their frozen-pack scores. 10 candidates. |
| `…/stratego_phase14_sidecar_eval/ANALYSIS.md` | **`ENGINEERING`, out-of-repo** | 22,000 games re-reading all ten candidates. **Explicitly not a Phase 14 artifact**, fed to no ledger, changed no frozen quantity. Its ordering (h=18 first) did **not** become a selection. |
| `reports/phase13/phase14_checkpoint_selection_rule_v1.json` | `ACCEPTED` — **never executed** | The frozen rule. Its `procedure_at_hour_168` was never reached. |
| `reports/phase13/phase14_launch_manifest_v1.json` | `ACCEPTED` — **currently the only tracked diff** | A required self-referential rebuild (it names itself in its own dirty list). **Do not commit, stash or revert it until Phase 14 is formally closed.** |
| `reports/phase14/phase14_dashboard_verification_report.md` | `ACCEPTED` | Read-only monitor, 62 tests. Two known defects (wedging keep-alive socket; PID discovery window) are **unfixable in-tree while a run is open**; `dashboard_pid_discovery_fix.patch` must stay out of the repo. |
| `reports/phase16/logs/` | `HISTORICAL` | Raw run logs and suite outputs. |

---

## 5. Operator / human-evaluation state

| Item | Status |
|---|---|
| `reports/phase16/operator_protocol_v1.md` | **`ENGINEERING`** — protocol delivered (10-game re-baseline + 20-game exam, pass = EWR ≥ 0.50) |
| `stratego/evaluation/phase16/operator_log.py`, `scripts/phase16_capture_setup.py`, `scripts/play_phase16_operator.py` | `ENGINEERING` — tooling delivered and exercised by tests |
| `data/phase16/operator_games.jsonl` | **`PENDING`** — **one line, and it is the CLI self-play verification game, not an operator game** |
| `operator_harvest` family | **`PENDING`** — present but empty, 0 setups |
| Operator re-baseline series | **`PENDING`** — 0 games played |
| Operator exam (`phase16_goal_v1`) | **`PENDING`** — never run |
| The original 85%-vs-casual-humans target | **`SUPERSEDED` / retired 2026-08-25** — **retired, not achieved** |
| Any prior informal human impressions | **`HISTORICAL`** — explicitly retired by Phase 16 |

**There is no human-strength evidence in this repository.** No internal-bot EWR
may be presented as a human win rate.

---

## 6. Repository-health items (recommendations only — nothing was changed)

### 6.1 Untracked Phase 15–16 work — the largest current risk

> **Amended 2026-08-31.** The same risk now extends to **Phase 17 and the Phase 18
> instruction package**, which are also entirely untracked: `reports/phase17/`,
> `data/phase17/`, `checkpoints/phase17/` (33.5 GB), `stratego/evaluation/phase17/`,
> `tests/evaluation/phase17/`, six `scripts/*phase17*` runners, and
> `instructions/phase_18_setup_integrated_warmstart/`. Phase 18 Agent 1 recorded
> and hashed every artifact it cites in
> [`../reports/phase18/phase18_process_boundary_v1.json`](../reports/phase18/phase18_process_boundary_v1.json)
> and changed nothing. **A Phase 18 training run cannot bind an immutable source
> closure until the operator either commits this work on a branch or signs an
> explicit dirty-list manifest.**
>
> **Resolved 2026-09-01 for Phase 17/18 only.** Phase 18 Agent 2 committed every
> path named above, except `checkpoints/phase17/`, on
> `phase18/setup-integrated-warmstart-g1` under P18-D001, and Gate G1 runs from a
> clean detached worktree at that closure. **Phase 15-16 is untouched and the risk
> below still stands in full.**

The entire Phase 15 and Phase 16 implementation and evidence base is
**untracked**: `stratego/belief/phase15/`, `stratego/search/phase15/`,
`stratego/search/phase16/`, `stratego/evaluation/phase16/`,
`stratego/training/phase16/`, the matching `tests/*/phase15|phase16/` trees,
nine `scripts/*phase15*|*phase16*` runners, `reports/phase15/`,
`reports/phase16/`, `data/phase15/`, `data/phase16/`,
`checkpoints/phase15/`, `checkpoints/phase16/`, and both phase instruction
directories.

This is **intentional and correct today** — the Phase 14 repository freeze
forbids committing anything, and lifting it is the not-yet-run Phase 16
Agent 5's first task. It is still a risk, and it must not be normalized.

**A backup archive is not a substitute for version-control history.** The tars
at `/Volumes/Brandon_Washington/stratego_untracked_backup_20260825.tar`
(8,137,132,544 bytes, sha256 `712f4c4a…`) and `…_20260826.tar`
(8,145,454,592 bytes, sha256 `18fd6e39…`) preserve *bytes at two instants*.
They carry no commit history, no diffs, no branches, no review trail, no
authorship, and no way to bisect. They live on the same external volume as the
Phase 14 run. They are disaster insurance, not source control.

**Nothing here was staged, committed, deleted or moved.**

### 6.2 `.gitignore` does not classify Phase 15–16 output

`.gitignore` carefully classifies Phase 8–14 generated bytes, and **stops
there**. It says nothing about `checkpoints/phase15/`, `checkpoints/phase16/`,
`data/phase15/` or `data/phase16/`.

**Recommendation (requires owner approval; the file was deliberately not
edited):** when the freeze lifts, add entries in the same style as the existing
ones — a comment naming what the bytes are and how to regenerate them — so that:

- the **large regenerable bytes stay out**: `checkpoints/phase15/*_prefix_*.npy`
  (~15 GB, already excluded from the backups as regenerable),
  `checkpoints/phase15/*_development_probabilities.npy`,
  `data/phase15/phase15_belief_corpus_v1/`, `checkpoints/phase16/arms/`;
- the **small identity-bearing artifacts go in**: `b18_belief_v1.pt`,
  `b24_belief_v1.pt`, `p18_source_identity.json`, `p24_source_identity.json`,
  the three `data/phase16/phase16_*.json` instrument files, and every
  `checkpoints/phase*/*_candidate_v1.json`;
- `checkpoints/phase16/COMPUTE_LOCK.json` is ignored (transient).

Decide `checkpoints/phase15/p18_source_readonly.pt` /
`p24_source_readonly.pt` deliberately: 3.5 MB each, mode `0444`, and they are
the only in-repo copies of models that otherwise exist **solely on the external
volume**. Tracking them is defensible.

### 6.3 Machine-specific absolute paths (portability dependency)

Absolute paths are embedded across the project — roughly **78 files** reference
`/Volumes/Brandon_Washington` and a similar number reference
`/Users/brandonwashington/…`, spanning `reports/`, `data/*_root.txt`,
`instructions/`, `scripts/` and a few modules under `stratego/`.

The load-bearing ones:

- **`/Volumes/Brandon_Washington/stratego_phase14/`** — the only copy of the
  interrupted run, its archive snapshots and its candidate ledger. **P18 and
  P24's source bytes live here.** If this volume is lost or renamed, the Phase 14
  evidence and both backbones' provenance go with it.
- `data/phase9_rollout_root.txt`, `data/phase10_corpus_root.txt`,
  `data/phase10_soak_root.txt`, `data/phase10b_rollout_root.txt`,
  `data/phase11_prediction_root.txt`, `data/warmstart_corpus_root.txt` — pointer
  files whose contents are absolute external paths.
- `monitoring/README.md`, `PHASE_14_RUNBOOK.md` — operational docs with the
  volume path baked in.

**These are identified, not fixed.** Historical artifacts record where things
actually were and must not be retro-edited. The recommendation for **future**
work is in [`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md) §4: resolve
storage roots through a single indirection and record the resolved absolute path
in the run's evidence rather than in code.

### 6.4 `stratego_project_docs 2/` — stale duplicate

A 10-file duplicate of this folder, last modified 2026-08-09, missing
`11_batch_simulation_spec.md` and `12_trajectory_buffer_spec.md`, with every
retained file smaller than its counterpart here.

**Status: `HISTORICAL` — stale and non-authoritative. Never read from it, never
write to it, never cite it.** It is already git-ignored. **It was not deleted or
moved**; removing it requires the project owner's explicit approval.

### 6.5 Out-of-repository consumer

A browser client at `../webapp3/` imports this repository by absolute path
(`_enginepath.py`, `neural.py`, `phase12.py`, `phase15.py`, `phase16.py`). It is
outside this repository, is not covered by this documentation set, and was not
audited. Two earlier siblings (`../webapp/`, `../webapp2/`) also exist.

### 6.6 Known live defects, recorded not repaired

| Defect | Where | Why not fixed |
|---|---|---|
| Human-play CLI draws setups through the mis-orienting Phase 11B glue | `scripts/play_phase12.py` (~209–250) | Repository freeze. `play_phase15.py` / `play_phase16.py` are unaffected. |
| Dashboard wedges permanently on a pooled keep-alive socket; PID discovery window scrolls the `launch` event out after ~66 h | `monitoring/phase14_dashboard/server.py` | Unfixable in-tree while a run is open; the fix patch must stay out of the repo. |
| `phase11_records.manifest_digest` embeds per-game wall-clock | `stratego/evaluation/phase11_records.py` | Deliberately not repaired inside a validation phase; no hard gate reads it. |

---

## 7. Phase 17 (tandem self-play) and Phase 18 artifacts

Added 2026-08-31. Nothing in this section was edited, moved or deleted.
**Updated 2026-09-01:** every path below except `checkpoints/phase17/` is now
committed on `phase18/setup-integrated-warmstart-g1`; the checkpoints remain
untracked by `.gitignore`, as intended.

### 7.1 Phase 17 — `RUN-2026-B`

| Artifact | Status | Notes |
|---|---|---|
| `checkpoints/phase17/RUN-2026-B/` | **`COMPLETE` — nothing promoted** | 25 paired candidates, all byte-verified between write time and post-termination. 33.5 GB unpruned. `joint_00535.pt` is **past the h12 boundary, was never exported, and must not be evaluated.** |
| `reports/phase17/agent_07_report.md`, `phase17_run_closeout_v1.json` | **`ACCEPTED` (as a run record)** | 12.658 active hours, 535/640 iterations, operator-terminated. Zero integrity events in all 535 rows. |
| `reports/phase17/agent_05_report.md`, `local_eval/` | **`ENGINEERING`** | Post-training evaluation, 2026-08-30. Bit-deterministic including across worker counts. |
| The Phase 17 *result* | **negative — see [`STATUS.md`](STATUS.md) §13** | Move-only degraded (t = −2.97); joint flat; 0 of 24 candidates beat hour 0; setup policy −0.0679 EWR below the fixed library. |
| `data/phase17/phase17_composite_benchmark_v1.json` | **`ENGINEERING` pack** | 120 boards = 10 opponents × 2 colours × 6 games. **Minimum detectable effect 0.138 EWR at 80% power** — see §7.3. |
| `reports/phase17/ataraxos_method_map_v1.md` | **`SUPERSEDED`** | Paper-only. Replaced by `reports/phase18/ataraxos_setup_method_map_v2.md`, which is checked against the authors' published implementation. Preserved unedited. |
| `stratego/evaluation/phase17/evaluator.py` | `ENGINEERING` — **carries a known defect** | A refusal receipt permanently blocks re-evaluating that candidate (`existing_result`, line 174). Deliberately not fixed, to keep one evaluator source digest across all 25 receipts. Required regression case for any future evaluator. |

### 7.2 Phase 18 — Agent 1 outputs

| Artifact | Status | Notes |
|---|---|---|
| `reports/phase18/phase18_process_boundary_v1.json` | `ACCEPTED` (as a record) | Working-tree classification, active-process state, storage, and digests of every cited artifact. Nothing modified. |
| `reports/phase18/phase18_phase8_reproduction_contract_v1.json` | `ACCEPTED` | All 12 frozen Phase 8 identities **independently recomputed**; the canonical fresh C1 initialization **reproduces bit-exactly from seed 2026081302**; all 28,000 corpus games re-hashed at payload level with zero mismatches. |
| `reports/phase18/ataraxos_setup_method_map_v2.md` / `.json` | `ACCEPTED` | 35 method elements against paper **and** the authors' published code at commit `92db29e8`. 22 `exact`, 6 `corrected`, 2 `scaled`, 4 `intentional integration divergence`, 1 `not used`. |
| `reports/phase18/phase18_evaluation_contract_v1.json` | `ACCEPTED` — **two packs unpopulated** | Lanes, pairing, metrics, bootstrap, anti-leakage rules, evaluator requirements, practical margin (0.05 EWR) and sample sizes, all frozen before any Phase 18 result. |
| `reports/phase18/agent_01_report.md`, `phase18_agent1_handoff_v1.json` | `ACCEPTED` | Findings and readiness booleans. |
| `reports/phase18/decisions/P18-D001.*` | `ACCEPTED` | The gate-G0 decision packet. |

### 7.2b Phase 18 — Gate G1 (Agents 2 and 3)

| Artifact | Status | Notes |
|---|---|---|
| `reports/phase18/phase18_g1_*` (control run, checkpoint manifest, noninferiority, arms, binding, launch) | `ACCEPTED` | Agent 2's G1 control at `G1_SOURCE_COMMIT 66b733ad`: 42/42 original gates, 7/8 paired margins; vs-random uncertifiable at 1,024 pairs (±0.0116 vs 0.010). |
| `reports/phase18/decisions/P18-D002.*`, `reviews/P18-D002_REVIEW.md` | `ACCEPTED` as `REVISE` | Authorized exactly one measurement-only revision: 4,096 independent pairs, same margin and rule. |
| `reports/phase18/phase18_g1_random_confirmation_*` (contract, bank, launch, reference, candidate, noninferiority, binding) + `g1_random_confirmation/` (receipts, run/arm records) | `ACCEPTED` | Agent 3's powered confirmation at `G1_CONFIRM_SOURCE_COMMIT 9392c6ec`: delta **+0.006348**, 95% **[+0.000793, +0.011902]** on 4,096 independent pairs vs the −0.010 margin — certified; zero integrity events; sealed-test access zero. Independently reproduced from the receipts by the review. |
| `reports/phase18/decisions/P18-D003.*`, `agent_03_report.md` | `ACCEPTED` as `PROCEED` — **G1 CLOSED** | Accepted 2026-09-02. Branch `phase18/g1-random-confirmation` published at `ef7523c1940650c0906d1927e64679e8328a663f` (local == remote, non-force). **Carries one narrative erratum**: `identity.rules` says battleless limit 100; the measurement used the evaluation value 200 (see the errata row). Packet not rewritten. |
| `reports/phase18/reviews/P18-D003_REVIEW.md` | `ACCEPTED` | Reviewing-chat audit: identities, receipts and the 4,096-pair result reproduced; authorizes G2 only. |
| `reports/phase18/phase18_rule_identity_errata_v1.json` | `ACCEPTED` (as a record) | E-P18-D003-RULES-1: the packet's 100 is a metadata error, the contract/engine/schedule/receipts carry 200, no rerun. O-P18-EVALRULES-1: `phase18_evaluation_contract_v1.json` names *training* rules (100) for future play lanes — **amendment required before any real-game G3/G4 evaluation**; does not affect G2. |

### 7.3 Two cross-cutting facts that change how earlier evidence reads

1. **The 120-board pack is underpowered for the questions it was used on.**
   Measured from Phase 17's own per-case paired outcomes: the within-candidate
   correlation between the two lanes on the same board is only **0.238**, so the
   paired difference SD (0.539) is *larger* than either lane's own SD (0.42–0.45).
   Pairing buys about a 24% variance reduction, not an order of magnitude. The
   minimum detectable effect is **0.138 EWR at n = 120**, and about **913 games**
   are needed for a 0.05 EWR margin. Per-opponent strata at n = 12 have an MDE near
   **0.44 EWR** and carry no information — Phase 17's reported worst-stratum
   figures should be read as descriptive only.

2. **The setup library is fully consumed by the accepted Phase 8 corpus.**
   `setup_library_v1` has 16 families × 500 bases split 400/50/50, the `neutral_v1`
   profile samples uniformly over all 16, and the Phase 8 corpus draws the library
   train/validation/test split for its own train/validation/test split respectively.
   **No library family is unseen**, and the Phase 16 `targeted_family` source is
   documented as *accepted library bases, family-targeted*, so it does not qualify
   either. Any future "unfamiliar opponent setup" claim needs **newly generated
   families**, not a held-out slice of the existing library.
