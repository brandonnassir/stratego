# Evidence and artifact-status index

**Written 2026-08-27.** A compact classification of every major artifact.
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
