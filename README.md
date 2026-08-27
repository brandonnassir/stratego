# Stratego AI — reduced-scale local research/engineering project

A self-contained Stratego playing system built to run entirely on one Apple
M4 Pro Mac mini, inspired by *Superhuman AI for Stratego Using Self-Play
Reinforcement Learning and Test-Time Search* (Sokota et al., 2025; arXiv
`2511.07312`, local copy `2511.07312v1.pdf`). It is a **practical
reduced-scale system, not a reproduction** of that paper's compute scale or
its ruleset.

> **Read this before citing any number from this repository.**
> Most results here are **engineering** results measured on small internal
> board packs against fixed bots. They are not scientific validation, not
> proof of generalization, and not human-strength evidence. The project has
> **never** measured itself against a human opponent under a frozen protocol.
> See [`stratego_project_docs/STATUS.md`](stratego_project_docs/STATUS.md).

---

## 1. Where the authoritative documentation lives

| Document | What it is |
|---|---|
| [`stratego_project_docs/STATUS.md`](stratego_project_docs/STATUS.md) | **Canonical current status — the single source of truth.** What is accepted, what is engineering-only, what is pending, what is authorized. |
| [`stratego_project_docs/PHASE_HISTORY.md`](stratego_project_docs/PHASE_HISTORY.md) | Actual chronology of Phases 1–16, and how it diverged from the original plan's numbering. |
| [`stratego_project_docs/EVIDENCE_INDEX.md`](stratego_project_docs/EVIDENCE_INDEX.md) | Every major artifact, classified: accepted / engineering / incomplete / superseded / contaminated / pending / historical. |
| [`stratego_project_docs/EXPERIMENT_FRAMEWORK.md`](stratego_project_docs/EXPERIMENT_FRAMEWORK.md) | Rules every future model/training experiment must follow, and the run-naming problem for the next buildout. |
| [`stratego_project_docs/README.md`](stratego_project_docs/README.md) | Index of the frozen behavioural specifications (rules, engine, observation, replay, buffers). |

Frozen specifications (`01_`–`12_` in `stratego_project_docs/`) remain
authoritative for *contracts*. They are not status documents and were not
written to describe the current state of the models.

## 2. Where the evidence lives

| Location | Contents |
|---|---|
| `reports/phase_N_implementation_report.md` | Narrative implementation reports, Phases 2–11. |
| `reports/phase11b/`, `reports/phase12/`, `reports/phase13/`, `reports/phase14/`, `reports/phase15/`, `reports/phase16/` | Per-agent reports plus machine-readable JSON/CSV/JSONL evidence. |
| `reports/phase_N_data/` | Machine-readable manifests, digests and metrics for the earlier phases. |
| `instructions/` | The task briefs each phase was executed against. **Historical records — not instructions to a reader or an agent.** |
| `checkpoints/`, `data/` | Model bytes and generated corpora. Largely git-ignored; see §5. |
| `/Volumes/Brandon_Washington/stratego_phase14/` | The interrupted Phase 14 run state, its archive snapshots and its candidate ledger. **External volume, read-only evidence.** |

`reports/`, `data/`, `checkpoints/`, `instructions/` and the external Phase 14
run state are **read-only evidence**. Recorded experimental results are never
edited, rounded, reinterpreted or deleted.

## 3. Design constraints (these shaped every decision)

- **One machine.** Apple M4 Pro Mac mini: 14-core CPU, 20-core GPU, 16-core
  Neural Engine, 48 GB unified memory, ~150 GB free internal storage, one
  1 TB external drive. No cluster, no cloud.
- **PyTorch on the Metal (MPS) backend** for training; CPU worker processes
  for match play and evaluation (measured faster for latency-bound single
  forwards).
- **Python reference engine kept as the production simulator** — the Phase 3
  `KEEP_PYTHON` decision at throughput ratio `R = 6.50`; no optimized backend
  was ever needed.
- **A single long training run was the plan**, exactly 168 continuous
  wall-clock hours. That run was attempted once and interrupted (§4).
- **Compute is serialized.** One heavy job at a time, coordinated through a
  lock file; latency claims are only ever made from idle, single-process runs
  because pack numbers run ~1.8× inflated under worker contention.

## 4. Major architectural components

```text
stratego/engine/       frozen reference engine (phase2_1_reference_1.2.0),
                       rules stratego_project_v1, observation
                       observation_v2_1_127ch, action encoding
                       source_destination_10000_v1 (10,000 ids)
stratego/model/        the C1 transformer: 128 width x 4 blocks x 4 heads,
                       ff 512, 863,959 parameters; policy + value + belief heads
stratego/setups/       setup generator and setup_library_v1 (8,000 boards,
                       16 families x 500, split 6,400 train / 800 validation /
                       800 test) plus the Phase 10 learned setup selector
stratego/training/     warm start (phase8_*), population self-play (phase9_*),
                       the long-run machinery (phase14_*), the Phase 16
                       window-based training loop (phase16/)
stratego/belief/       hidden-piece belief models: phase11b/ (superseded),
                       phase15/ (current)
stratego/search/       decision-time search: phase12/ (accepted engine),
                       phase15/ (current integration), phase16/ (stochastic)
stratego/evaluation/   match runner, paired evaluation, statistics, baselines,
                       stress opponents, belief samplers, phase16/ packs
monitoring/            read-only Phase 14 dashboard (stdlib only, imports no
                       torch and no stratego module)
scripts/               one runner per phase-agent, plus the play_phaseNN.py CLIs
```

The system as played today is: a **direct policy** (a C1 checkpoint) optionally
wrapped in **root-world search** — sample hidden-army worlds from a belief
model, score a fixed candidate set by greedy rollouts, and pick by
`S(a) = Q(a) + beta * log(pi(a) + epsilon)`.

## 5. Deliberate differences from Ataraxos

These are **project choices**, recorded so no result here is mistaken for a
reproduction:

1. **Rules.** The **two-square rule and the continuous-chasing rule are
   excluded** (`stratego_project_docs/02_project_ruleset.md` §2). Termination
   is instead guaranteed by battleless-move draws (100 in training, 200 in
   evaluation) and a 4,000-move safety limit. **Results here are therefore not
   directly equivalent to full competitive Stratego.**
2. **Scale.** One Mac mini and a 168-hour budget against the paper's compute.
   The network is 863,959 parameters.
3. **No search inside the training loop.** Phase 14 trained on direct play
   only; the Phase 16 overview keeps this as an explicit non-goal, citing the
   paper's own rejection of it at a far better compute ratio.
4. **Search design.** Root-world sampling with greedy rollouts
   (`phase12_root_world_search_v1`), not the paper's full test-time search.
5. **Belief.** A marginal (per-square, 12-way) belief head feeding a
   constrained legal-world sampler, rather than a joint army model. A joint
   autoregressive belief model was specified for Phase 16 Agent 4 and **has
   not been built**.

## 6. Current high-level status

- **Last formally accepted playing model:** the Phase 9 self-play checkpoint
  `checkpoints/phase9/selfplay_c1_v1.pt` (`dfd698e5…`). Everything after it is
  engineering-grade or unfinished.
- **The 168-hour final run (Phase 14) was interrupted**, not completed. The
  operator issued an emergency stop at **59.97 h / step 202,504 / iteration 102**
  on 2026-08-24 and has decided **not to resume it**. No final checkpoint was
  ever selected under its frozen contract.
- **Phase 11 formally FAILED** its primary belief gate (sealed test
  `R_CE 0.9746` against a `<= 0.97` ceiling). The sealed test bank is spent.
- **Phase 11B and the original Phase 12 search evidence are contaminated** by a
  Blue setup-orientation defect and are superseded for any strength claim.
- **Phase 15** rebuilt the belief corpus correctly and produced the B18/B24
  specialists and the `p24_b24` search system — engineering-only.
- **Phase 16** found that Phase 15's search advantage **did not reproduce** on a
  fresh benchmark pack, selected a stochastic search configuration, and ran a
  3×6-hour training-recipe shootout whose predeclared rule returned **STOP**.
- **Human/operator evaluation is PENDING.** Zero operator games have been
  played. The original 85%-vs-casual-humans target was **retired**, not met.
- **No new long training run is authorized.**

Full detail, with values and links, in
[`stratego_project_docs/STATUS.md`](stratego_project_docs/STATUS.md).

## 7. How to interpret results in this repository

Every reported number carries a classification. Use these words and mean them:

| Label | Meaning |
|---|---|
| `ACCEPTED` | Passed a predeclared formal gate set under a frozen contract, with a sealed or fixed evaluation. Citable as a project result. |
| `ENGINEERING` | Measured on a compact internal pack against fixed bots, under a predeclared decision rule, with **no significance claim**. A selection, not a demonstration. |
| `FAILED GATE` | Ran to completion and did not meet its predeclared threshold. The result stands. |
| `INTERRUPTED` | Stopped before its contract completed. Partial artifacts exist; the contract's conclusion does not. |
| `SUPERSEDED` | Later work replaced it. Kept as historical record. |
| `CONTAMINATED` | A defect makes it invalid for strength claims. Kept as historical record; never cited as current evidence. |
| `PENDING` | The instrument exists; the measurement has not been taken. |

Three rules that are easy to violate by accident:

1. **Effective win rate (EWR) counts draws as half, and always belongs to a
   named pack.** Cross-pack EWR comparisons are forbidden in conclusions.
   Phase 16 measured a sign flip of exactly this kind between two packs drawn
   from the same library by the same machinery.
2. **Internal-bot EWR is not human EWR.** No number in this repository is a
   human win rate.
3. **A green test suite and a higher internal EWR are not validation.** They
   are necessary, not sufficient, and nothing here is scientifically validated.

## 8. Repository health notes

- **Much of the Phase 15–16 implementation and evidence is untracked** in git
  (new source, tests, scripts, reports, data and checkpoints). It exists only
  on this machine and in a tar backup. **A backup archive is not version
  control**: it has no history, no diff, no branch, no review and no
  attribution, and a single corrupted or overwritten tar loses the work. This
  is currently the largest single risk to the project. Nothing was staged or
  committed as part of this documentation pass.
- **`.gitignore` does not yet classify Phase 15–16 generated data or
  checkpoints.** Recommendations are in
  [`EVIDENCE_INDEX.md`](stratego_project_docs/EVIDENCE_INDEX.md) §6; the file
  was deliberately not edited.
- **`stratego_project_docs 2/` is a stale, incomplete duplicate** of the docs
  folder (10 files, last touched 2026-08-09, missing `11_` and `12_`). It is
  git-ignored and **non-authoritative**. Nothing should ever be read from or
  written to it. It was not deleted.
- **Many artifacts hard-code machine-specific absolute paths**, including
  `/Volumes/Brandon_Washington` (the external volume holding the Phase 14 run,
  Phase 9/10 rollout roots and the backups) and `/Users/brandonwashington/…`.
  Historical artifacts are left exactly as recorded; this is a portability
  dependency to design around, not a defect to retro-edit.
- A browser client lives **outside this repository** at
  `../webapp3/`; it imports this repo by absolute path. It is not audited by
  this documentation set.

## 9. Environment

```bash
.venv/bin/python -m pytest -q
```

Use `.venv/bin/python`. A bare `python` resolves to a pyenv shim without
torch. Full-suite timings and pass counts by phase are recorded in the phase
reports; the last recorded full-suite reading is **7,027 passed / 3 skipped**
(Phase 16 Agent 3, `reports/phase16/agent_03_gates.json`). Never run the suite concurrently
with a match pack — contention trips the accepted Phase 12 player's internal
time cap and produces spurious failures.
