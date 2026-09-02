# Phase chronology — what actually happened, Phases 1–18

**Written 2026-08-27.** This replaces the "Phase 5 — Next" presentation that
[`README.md`](README.md) and [`05_project_plan.md`](05_project_plan.md) carried
until now. It is a **summary with links**, not a rewrite: the historical reports
are authoritative for their own contents and were not edited.

Status labels are defined in [`STATUS.md`](STATUS.md) §1 and the root
[`README.md`](../README.md) §7.

---

## 1. Planned numbering vs. numbering actually used

The original plan (`05_project_plan.md` §§13) allocated Phases 13–17 one way.
Execution diverged from Phase 13 onward. **The numbers were reused for different
work. They are not being retroactively renamed.**

| Number | Original plan | What that number was actually used for | Was the planned work done? |
|---|---|---|---|
| 13 | Integrated rehearsal | Integrated rehearsal **plus** the configuration freeze and the launch package | Yes — and it absorbed the planned Phase 14 |
| 14 | Configuration freeze | **The attempted final 168-hour run** (the planned Phase 15's job) | The freeze happened in Phase 13; the run was `INTERRUPTED` at 59.97 h |
| 15 | Final 168-hour run | **Corrective belief/search engineering** (unplanned; a response to the orientation defect and to Phase 11's failure) | No — the final run was Phase 14 and it did not complete |
| 16 | Automated final evaluation | **Robustness and distribution engineering** (unplanned) | Partly — `phase16_benchmark_v1` is a canonical automated instrument, but it is an engineering pack, not the planned post-run final evaluation |
| 17 | Casual human evaluation | **Two different things.** The planned human evaluation was never reached; the number was then reused for **tandem current-policy self-play** (`RUN-2026-B`), which ran and produced a negative result | No — the human evaluation is still `PENDING`, never run |
| 18 | *(not in the original plan)* | **Setup-integrated Phase 8 warmstart** — planned 2026-08-31, only Agent 1 authorized | Not started beyond Agent 1 |

Consequences a reader must hold onto:

- **"Phase 14" now means the interrupted run**, not a configuration freeze. Any
  future restart must not reuse the bare name — see
  [`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md) §6.
- **There was never a completed "final 168-hour run"** under any number.
- **There was never a human-evaluation phase.**
- **"Phase 17" is ambiguous and must always be qualified.** Written unqualified
  it is read as the planned human evaluation (never run). The executed work under
  that number is the **tandem self-play** phase, §13 below.

---

## 2. Phases 1–8 — contracts, engine, evaluation, model, warm start

### Phase 1 — Rules, observation, internal state, replay, validation contracts
- **Purpose:** freeze the behavioural contracts before any code.
- **Outcome:** `stratego_project_v1` (rules), `observation_v2_1_127ch`,
  `source_destination_10000_v1` (10,000 action ids), internal-state and replay
  schemas. **The two-square and continuous-chasing rules were deliberately
  excluded** (`02_project_ruleset.md` §2).
- **Status:** `ACCEPTED`. Still the governing contracts.
- **Evidence:** [`01_official_rules.md`](01_official_rules.md),
  [`02_project_ruleset.md`](02_project_ruleset.md),
  [`06_observation_v2_127ch.md`](06_observation_v2_127ch.md),
  [`08_internal_state_spec.md`](08_internal_state_spec.md),
  [`09_public_event_and_replay_schema.md`](09_public_event_and_replay_schema.md).

### Phase 2 / 2.1 — Python reference engine
- **Purpose:** a readable, correct reference simulator.
- **Outcome:** `PASS` — engine frozen as `phase2_1_reference_1.1.0` after the
  perspective-symmetry and terminal-precedence corrections.
- **Status:** `ACCEPTED`, later **superseded in version** by
  `phase2_1_reference_1.2.0` (Phase 6B correction). Rules version unchanged.
- **Evidence:** [`reports/phase_2_implementation_report.md`](../reports/phase_2_implementation_report.md).

### Phase 3 — High-throughput training architecture
- **Purpose:** decide whether the Python engine could carry production.
- **Outcome:** `PASS`. Backend decision **`KEEP_PYTHON`** at throughput ratio
  **`R = 6.50`**; the optimized-backend agent was never required.
- **Status:** `ACCEPTED`. The decision held for the whole project.
- **Caveat recorded at the time:** `R = 6.50` used an untrained representative
  probe, not the final architecture.
- **Evidence:** [`reports/phase_3_implementation_report.md`](../reports/phase_3_implementation_report.md).

### Phase 4 — Baseline opponents and the evaluation harness
- **Purpose:** build the thing every later number is measured with.
- **Outcome:** `PASS`. `policy_interface_v1`, `match_spec_v1`,
  `evaluation_setup_bank_v1` (1,024 pairs), `color_swap_same_board` pairing, the
  four-tier baseline ladder, six stress opponents, paired bootstrap statistics.
  100,000 hidden-information permutation trials / 1,000,000 comparisons /
  **0 mismatches**; 44,544-game calibration league / **0 illegal actions**.
- **Status:** `ACCEPTED`. Still the evaluation stack.
- **Evidence:** [`reports/phase_4_implementation_report.md`](../reports/phase_4_implementation_report.md).

### Phase 5 — Neural model contract and end-to-end integration
- **Purpose:** confirm the frozen representation drives a real PyTorch model.
- **Outcome:** `PASS`, 22/22 hard gates. `model_contract_v1`;
  `integration_model_v1` is an **integration fixture, not the production model**.
- **Status:** `ACCEPTED`.
- **Evidence:** [`reports/phase_5_implementation_report.md`](../reports/phase_5_implementation_report.md).

### Phase 6 / 6B — Production architecture selection and the engine correction
- **Purpose:** pick the network and re-measure throughput for real models.
- **Outcome:** **C1 selected** — 128 width × 4 blocks × 4 heads, ff 512,
  **863,959 parameters**. Phase 6B applied an authorized correctness correction
  advancing the engine to `phase2_1_reference_1.2.0`.
- **Status:** `ACCEPTED`. Agents 1–6 ran against `1.1.0` and are the historical
  record; the §6B-3.3 differential validation connects the two versions.
- **Evidence:** [`reports/phase_6_implementation_report.md`](../reports/phase_6_implementation_report.md).

### Phase 7 — Setup generator and setup library
- **Purpose:** a diverse, reproducible setup distribution.
- **Outcome:** `PASS`, 28/28 gates. **`setup_library_v1`: 8,000 boards**,
  16 families × 500, split 6,400 train / 800 validation / 800 test, content
  digest `7b8a666…`.
- **Status:** `ACCEPTED`. Still the library everything draws from.
- **Evidence:** [`reports/phase_7_implementation_report.md`](../reports/phase_7_implementation_report.md),
  `data/setups/setup_library_v1_manifest.json`.

### Phase 8 — Synthetic warm start
- **Purpose:** warm-start C1 from rule-based play before self-play RL.
- **Outcome:** `PASS`, 42/42 gates at Agent 7's held-out evaluation. One
  canonical run, 25,000 optimizer updates from a fresh C1 initialisation; the
  checkpoint was selected by validation alone, with instrumented proof that
  0 test examples reached any model.
- **Status:** `ACCEPTED`. Its output is the **Phase 8 anchor** used as an
  evaluation opponent for the rest of the project.
- **Evidence:** [`reports/phase_8_implementation_report.md`](../reports/phase_8_implementation_report.md).

---

## 3. Phase 9 — Population self-play  ·  the last accepted policy

- **Purpose:** self-play RL over a population, from the Phase 8 warm start.
- **Outcome:** **`PASS`, all 8 hard gates.** Sealed final evaluation:
  **vs-anchor EWR 0.8442**, Strategic **+0.366**, Tactical **+0.370**. Frozen at
  iteration 40 / `behavior_B041.pt`; the observer reconciliation at iteration 30
  matched exactly (245,490).
- **Frozen artifact:** `checkpoints/phase9/selfplay_c1_v1.pt`
  (`dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea`).
- **Status:** **`ACCEPTED` — and still the latest accepted direct policy.**
  Nothing later replaced it.
- **Superseded?** No.
- **Evidence:** [`reports/phase_9_implementation_report.md`](../reports/phase_9_implementation_report.md);
  closed at commit `427b963`.

---

## 4. Phase 10 / 10B — Learned setup selection

### Phase 10
- **Purpose:** learn which setups to play, rather than sampling uniformly.
- **Outcome:** **`PASS-NONINFERIOR`**, 8/8 hard gates. Direct reading
  **0.5151** with paired 95% LB **0.4883**; league **Δ_L +0.0125** (LB −0.0041).
  11,264 games. `phase10_setup_selector_v1` (`5e2b9c3a…`).
- **Status:** `ACCEPTED`. Formally closed at commit `17188a5`.
- **Read it honestly:** non-inferior, not superior.
- **Evidence:** [`reports/phase_10_implementation_report.md`](../reports/phase_10_implementation_report.md).

### Phase 10B — optional setup-conditioned fine-tuning
- **Purpose:** advisory experiment; explicitly optional.
- **Outcome:** **`INCOMPLETE`**. Paused by operator request at a clean resumable
  boundary after **5 of 30 iterations** (10,240 of 61,440 games, 9,074 optimizer
  steps, 2.00 h of a 12 h ceiling). **No classification.** The sealed test bank
  was never opened. *"unfinished, not failed."*
- **Status:** `INCOMPLETE`. Never resumed.
- **Evidence:** [`reports/phase_10b_implementation_report.md`](../reports/phase_10b_implementation_report.md).

---

## 5. Phase 11 — Belief validation  ·  FAILED GATE

- **Purpose:** decide, under a sealed test, whether the belief head was good
  enough to authorize decision-time search.
- **Outcome:** **`FAIL`.** The first and only sealed evaluation of
  `phase11_test_bank_v1` (2,048 paired cases / 4,096 games) returned
  **`R_CE 0.9746` [0.9726, 0.9764]** against Gate A's `<= 0.97` ceiling.
  Gates B–H all passed (Δtop-1 **+0.0352**, ECE 0.0409, all safety counters
  zero, 8/8 topology legs exact, p95 forward+64 worlds 48.51 ms).
- **Classification recomputed from the gate rows alone: `FAIL`. Phase 12 was
  not authorized by Phase 11.**
- **The risk was known and deliberately not repaired:** the validation reading
  `R_CE 0.9750` was carried into the sealed run unchanged — no calibration,
  threshold, bin, baseline, bank, stratum or sampler rule moved before or after.
  A validation phase was not allowed to become a repair loop.
- **The sealed test bank is permanently spent.** A future belief-repair phase
  needs fresh sealed evidence.
- **Status:** `FAILED GATE`. The result stands. Committed at `3a0720c`.
- **Known defect not repaired:** `phase11_records.manifest_digest` embeds
  per-game `forward_seconds`, so two executions of one bank cannot agree on it.
  Cross-run identity is the `store_content_digest`; no hard gate reads the
  manifest digest.
- **Evidence:** [`reports/phase_11_implementation_report.md`](../reports/phase_11_implementation_report.md) §7.

---

## 6. Phase 11B — Belief engineering sprint  ·  CONTAMINATED

- **Purpose:** an engineering sprint to find a better belief model after
  Phase 11's failure, on a common corpus.
- **Outcome:** a leaderboard, and a selection made **for engineering reasons
  with `scientific_claim = none`**: Agent 1C (`R_CE 0.9460`, top-1 0.2640).
  Leaderboard order as recorded: 1C 0.9460 < 1B 0.9495 < 1A 0.9531 <
  A4 0.9614 < A3 0.9624 < A2 0.9686 < old head 0.9834. Agent 4 established that
  **raw and C1 features are not complementary**; Agent 5 (autoregressive
  transformer) was **cancelled**, not attempted. The sprint's own conclusion was
  that **corpus, not architecture, is the binding constraint**.
- **Status:** **`CONTAMINATED` for any strength or belief-quality claim.**
  `phase11b_common_corpus_v1` carries the Blue setup-orientation defect
  ([`STATUS.md`](STATUS.md) §9), so every metric measured on it is invalid as
  current evidence. Kept as historical record.
- **Superseded by:** Phase 15's corrected corpus and the B18/B24 specialists.
  Re-scored on the corrected corpus, Agent 1C collapses to **`R_CE 0.9996`**
  [0.9957, 1.0030] — indistinguishable from the count baseline. *That drop is
  not a clean measurement of the orientation cost*: backbone and distribution
  changed at the same time.
- **Evidence:** [`reports/phase11b/phase11b_closure_report.md`](../reports/phase11b/phase11b_closure_report.md),
  [`reports/phase11b/phase11b_engineering_selection_v1.json`](../reports/phase11b/phase11b_engineering_selection_v1.json).
- **Caution:** a repeat-training pass overwrote the 1B/1C first-pass checkpoints.
  Bind to surviving bytes, not to report text.

---

## 7. Phase 12 — Decision-time search  ·  engine accepted, evidence contaminated

- **Purpose:** benchmark rollout count, depth and regularization for a practical
  human-play time budget.
- **Outcome:** the search engine **`phase12_root_world_search_v1`** — root-world
  sampling, fixed candidate set, greedy rollouts,
  `S(a) = Q(a) + beta·log(pi(a) + epsilon)` — plus the project's first working
  player, `phase12_search_player_v1` (modes direct/tiny/small/medium, TINY
  default at 0.126 s/move). Search beat direct C1 by **+0.10 to +0.16 EWR** on
  its pack, but **neither belief choice nor budget separated** within the 0.10
  engineering margin. MEDIUM was designated maximum-strength by operator
  direction (EWR **0.6875**, 0.846 s/move).
- **Status:** split, and the split matters:
  - the **search engine** is reused unmodified by Phases 15 and 16 — treat it as
    accepted infrastructure;
  - the **match evidence and every Phase 12 EWR** are **`CONTAMINATED`** by the
    orientation defect (47 of 64 boards observed with front-row flags) and must
    never be cited as current strength evidence;
  - `phase12_search_candidate_v1` is **`SUPERSEDED`** as a candidate.
- **Still-live defect:** `scripts/play_phase12.py` continues to draw human
  setups through the mis-orienting Phase 11B glue. Found, reported, deliberately
  **not** fixed under the repository freeze.
- **Evidence:** [`reports/phase12/`](../reports/phase12/), especially
  [`agent_05_report.md`](../reports/phase12/agent_05_report.md).
- Phase 12 pipeline complete and committed at `6559460`.

---

## 8. Phase 13 — Final training integration, freeze, and launch package

- **Purpose:** the planned "integrated rehearsal" — **plus** the planned
  Phase 14 configuration freeze, which was folded in here.
- **Outcome:** all four agents `PASS`; **Agent 4 returned GO (Gates A–J, 90/90)**
  and froze the Phase 14 launch package.
  - contract `phase13_final_training_contract_v1` frozen;
  - runner built, `integrated_config_digest 9c2a38e4…`;
  - a real **90-minute rehearsal at the frozen 2,048-game production
    population** survived a process-group `SIGKILL` and a CPU-worker kill,
    stopped at its own deadline, **16/16 readiness**, `rehearsal_digest
    d8ebae4e…`;
  - measured **~21 min/iteration** and **0.2024 GiB/h** (34.8 GiB per 168 h);
  - **one real defect found and fixed**: a killed loader worker used to kill the
    learner (`BrokenProcessPool` was in neither error list).
- **Status:** `ACCEPTED`. **Formally accepted — do not rerun the rehearsal or
  the worker-kill verification.**
- **Superseded?** Its *plan* was, by events: the run it launched did not finish.
  Its artifacts are intact.
- **Evidence:** [`reports/phase13/`](../reports/phase13/),
  [`PHASE_14_RUNBOOK.md`](../PHASE_14_RUNBOOK.md),
  [`phase14_launch_manifest_v1.json`](../reports/phase13/phase14_launch_manifest_v1.json).
  Dashboard committed at `124f3be`.

---

## 9. Phase 14 — The attempted final 168-hour run  ·  INTERRUPTED

- **Purpose:** the single long training run the whole project was built for.
- **Outcome:** **stopped by operator emergency stop at 59.97 h / step 202,504 /
  iteration 102 on 2026-08-24T04:19Z.** 10 of a planned 29 candidates exist
  (hours 0–54). `closed: false`. **No checkpoint was selected under the frozen
  contract**, because that contract's procedure begins at hour 168.
- **Status:** **`INTERRUPTED`. Never describe it as completed.** The operator has
  decided it will **not** be resumed; formal closure is blocked until
  **2026-08-28T16:15:34.689Z**.
- **What it established anyway:**
  - all the measurable learning happened in the **first six hours**
    (h0 0.7600 → h6 0.8014 = **+0.0414**, p < 0.0001; h6 → h54 trends
    **−0.0342 EWR / 100 h**);
  - **83% of wall-clock was training, 17% collection**; minutes/iteration grew
    15.7 → 153.8 because iterations were sized in **whole games**;
  - the frozen 128-game pack is **noisy, not inflated** (mean delta +0.0066 at
    2,200 games) — never apply a blanket correction;
  - stop latency is phase-dependent: **75 s from COLLECTING, 129 min from
    TRAINING** (no stop check inside training).
- **Also delivered:** a read-only monitoring dashboard (`monitoring/`,
  pure-stdlib, imports no torch and no `stratego` module). **Two known,
  unfixable-in-tree defects:** a single-threaded HTTP/1.1 server with no handler
  timeout that wedges permanently on a pooled keep-alive socket (closing the tab
  does not free it — only a restart does), and PID discovery that scans a
  400-record window so the `launch` event scrolls out after ~66 h.
- **Evidence:** `/Volumes/Brandon_Washington/stratego_phase14/` (run state,
  archive, candidate ledger, emergency-stop record);
  `/Volumes/Brandon_Washington/stratego_phase14_sidecar_eval/ANALYSIS.md`
  (the 22,000-game re-evaluation — **not a Phase 14 artifact**, fed to no
  ledger); [`reports/phase14/`](../reports/phase14/) (dashboard verification).

---

## 10. Phase 15 — Corrective belief and search engineering  ·  ENGINEERING

Unplanned. A response to Phase 11's failure, the orientation defect, and the
interrupted run. Nothing from Phase 15 is committed to git.

### Agent 1 — clean corpus, B18/B24
- **Purpose:** rebuild the belief corpus without the orientation defect and
  train belief specialists on the two Phase 14 backbones.
- **Outcome:** the defect **quantified** (1.77% front-row flags corrected vs
  77.00% under the old glue, over 4,096 paired boards);
  `phase15_belief_corpus_v1` (155,027 positions / 4,373,492 hidden pieces,
  splits provably disjoint); **B18 `R_CE 0.9189`**, **B24 `R_CE 0.9172`**, both
  clearly beating the count baseline, with the old Agent 1C at **0.9996**.
- **Method lesson recorded:** calibration and development need a **per-family
  split of the validation library**, not just different seeds, or they share
  ply-0 positions. CPU beat MPS for both corpus generation and this training.
- **Status:** `ENGINEERING`. Established a corpus and two models; **no
  playing-strength claim**.
- **Evidence:** [`reports/phase15/agent_01_report.md`](../reports/phase15/agent_01_report.md).

### Agent 2 — search integration, ladder, deep pilot, mixture pilot
- **Purpose:** put the specialists behind the accepted search engine and choose
  a system.
- **Outcome:** **`p24_b24` at TINY** selected (max strength MEDIUM, caps
  0.91 / 5.0 s), reusing the Phase 12 engine unmodified. Search beat direct on
  this pack (**p24_b24 +0.1375 ± 0.0414**). **But the learned belief never
  separated from the `remaining_count` control**, and the oracle ceiling is only
  **+0.100 / +0.146 EWR**, moving 10–12% of decisions.
- **Deeper-search pilot:** 2.19× and 3.90× more compute made it **worse**
  (**−0.075 paired at both LARGE and XLARGE**) while the **oracle at the same
  budgets improved** (+0.042 / +0.025). **The world distribution, not the search
  mechanics, is what fails at scale.** XLARGE is also unshippable (idle p95
  6.989 s > 5 s). **Ladder closed.**
- **Mixture pilot:** closed at Stage 1, **no useful mixture**. Oracle Q-regret
  could not even distinguish `b24@MEDIUM` from `b24@LARGE` (+0.0008 ± 0.0017,
  110/120 tied), and the regret metric has a non-zero floor (0.0458).
- **Status:** `ENGINEERING`. **The +0.1375 search advantage did not reproduce in
  Phase 16** ([`STATUS.md`](STATUS.md) §8) — both readings stand.
- **Measurement lesson:** measure latency **idle, not in-pack** (~1.8× contention
  inflation).
- **Evidence:** [`agent_02_report.md`](../reports/phase15/agent_02_report.md),
  [`agent_02_deep_report.md`](../reports/phase15/agent_02_deep_report.md),
  [`agent_02_mixture_report.md`](../reports/phase15/agent_02_mixture_report.md),
  [`phase15_search_handoff_v1.json`](../reports/phase15/phase15_search_handoff_v1.json).

---

## 11. Phase 16 — Robustness and distribution  ·  ENGINEERING, partially executed

Unplanned; approved 2026-08-25 after an Ataraxos gap review. **The 85% target
was formally retired here** and replaced by `phase16_goal_v1` (EWR ≥ 0.50 over a
20-game operator exam under rematch conditions). Five agents were briefed;
**three ran.** Nothing from Phase 16 is committed to git.

### Agent 1 — measurement instruments and the operator protocol  ·  RAN
- Delivered `phase16_benchmark_v1` (120 frozen paired boards, digest
  `ebd13019…`, predeclared `quick60` subset), `phase16_adversarial_setups_v1`
  (9 families / 96 gate-validated setups), `phase16_adversarial_baseline_v1`,
  the operator protocol, logging schema, capture and harvest tooling.
- **Headline finding: the Phase 15 search advantage does not appear on this
  pack** (TINY − direct **−0.029 ± 0.036**; MEDIUM − direct **−0.017 ± 0.033**).
- **Second finding: adverse setups alone cost little overall** (0.0625 TINY /
  0.0469 MEDIUM) **and the cost is concentrated** in `spy_shadow` and the
  bombed-flag families.
- **The operator was unavailable: `operator_harvest` is empty and the
  re-baseline series is `PENDING`.**
- **Status:** `ENGINEERING`; instruments are frozen and executable.

### Agent 2 — stochastic search  ·  RAN
- Delivered two seed-deterministic knobs over the frozen engine and selected
  **`stoch_t015_r100`** (τ 0.15, τ_r 1.0, top-p 0.9), shipping as
  `varied_strength` (MEDIUM) and `varied_fast` (TINY), plus
  `scripts/play_phase16.py`.
- At τ=0, τ_r=0 it replays Phase 15's decisions **bit-identically**.
- **Status:** `ENGINEERING` — **the current candidate**. Selected for lowest
  repeat rate among arms not worse than the control, **not** for strength
  (+0.0167 ± 0.0640 at MEDIUM, −0.0167 ± 0.0594 at TINY).

### Agent 3 — training loop v2 and the 3×6-hour recipe shootout  ·  RAN
- Built a window-based collector (fixed decision budget per iteration), damped
  schedules, EMA, pure self-play; ran three 6-hour arms.
- **Verdict: `STOP`** — neither damped arm cleared the predeclared +0.03 margin;
  **no long run is authorized.**
- **Read it correctly: the experiment could not tell the three arms apart.**
  Every arm's curve sits inside one SE of its start; the 0.03 margin is
  **0.53 SE**; the same games over the full pack **reverse** the verdict.
- **What is solid:** iteration wall-time CV near 0.05 across ~300 iterations,
  0 vetoes, 0 non-finite losses, window-edge invariant exact (0.000000000).
  **Collection is a wash (0.90×); the win is pinned iteration sizing.**
- **A brief defect it caught and fixed:** the brief transcribed Ataraxos's LR
  schedule constants (fitted to ~43,000 iterations) into a ~313-iteration run,
  which would have floored the LR at n≈9 and run arms B/C **5× below the
  control**. Amended to `n_ref = ceil(0.125·N) = 40`, author-confirmed; entropy
  deliberately left alone; arm A untouched. **Never transcribe a paper's
  schedule constants without re-fitting them to your own iteration count.**
- **Status:** `ENGINEERING`. `phase16_recipe_candidate_v1` records
  `adopt_recipe.pass = False`.

### Agent 4 — joint/autoregressive belief worlds  ·  NOT RUN
- Specified (`04_AGENT_4_JOINT_BELIEF_WORLDS.md`); **never built**. There is no
  `stratego/belief/phase16/`. The deep-ladder rerun it gates remains closed.
- **Status:** `PENDING`.

### Agent 5 — closeout, production, operator exam  ·  NOT RUN
- Blocked by design until after **2026-08-28T16:15:34Z**, and additionally gated
  on Agents 1–4. Phase 14 formal closure, the git commits, the production run
  and the operator exam all belong to it.
- **Status:** `PENDING`. **This is why the repository freeze is still in force
  and why Phase 15–16 work is still untracked.**

- **Evidence:** [`reports/phase16/`](../reports/phase16/),
  [`instructions/phase_16_robustness_and_distribution/00_PHASE_16_OVERVIEW.md`](../instructions/phase_16_robustness_and_distribution/00_PHASE_16_OVERVIEW.md).

---

## 12. Phase 17 — Casual human evaluation  ·  NEVER REACHED

- **Purpose (as planned):** measure effective win rate against casual human
  players; the project's primary success criterion.
- **Outcome:** **never run.** The 85% target was retired 2026-08-25 for lack of
  a measurable human pool. Its replacement — a 20-game operator exam at
  EWR ≥ 0.50 — has also never been run: **zero operator games exist.**
- **Status:** `PENDING`.
- **Therefore:** the project has **not** demonstrated its stated 85% effective
  win rate against casual humans, and has no human-strength evidence of any
  kind.

---

## 13. Phase 17 (tandem self-play) — `RUN-2026-B`  ·  COMPLETE, NEGATIVE

Distinct from §12. This is the work actually executed under the number 17.

- **Purpose:** train a move policy and an autoregressive setup policy together
  by current-policy self-play, following the paper's setup-learning recipe as
  read **from the paper alone** — the authors' implementation was not available
  when the method map was written.
- **Run:** `RUN-2026-B`, launched from `90278aa`, **12.658 active hours, 535 of
  a frozen 640 iterations**, terminated by the operator after the twelfth hour.
  25 paired candidates exported, all verified byte-identical between write time
  and post-termination. Zero integrity events in all 535 rows.
- **Post-training evaluation (2026-08-30):** every candidate evaluated locally
  on the 120-board `phase17_composite_benchmark_v1` pack, both lanes, zero
  refusals, bit-deterministic including across worker counts.
- **Result — negative:**
  - move-only **degraded** over hours 6–12 (slope −0.0115/h, t = −2.97);
  - the joint lane was **flat** (t = 0.04);
  - **0 of 24 trained candidates beat the hour-0 Phase 9 C1 start**; the
    move-only curve peaks at hour 0;
  - the trained setup policy stayed **−0.0679 EWR below the fixed setup
    library** (t = −2.91) and never beat its own random initialization
    (difference-in-differences +0.0237, t = +0.44).
- **Status:** `COMPLETE`. **No checkpoint was promoted.** The accepted direct
  policy is unchanged — it is still Phase 9 `selfplay_c1_v1.pt`.
- **What it does and does not establish.** It is a valid negative result *for
  its exact implementation*. It does **not** establish that the paper's setup
  method fails here, for two reasons now documented in
  [`../reports/phase18/ataraxos_setup_method_map_v2.md`](../reports/phase18/ataraxos_setup_method_map_v2.md):
  the authors' published code differs materially from the paper-only reading
  Phase 17 implemented (entropy units, forced handedness, reusable setup pools,
  effective batch size); and the 120-board evaluation lane has a **minimum
  detectable effect of 0.138 EWR**, so it could not have resolved the effect
  sizes at issue.
- **Preserved, not promoted.** All Phase 17 evidence is intact and unedited in
  `reports/phase17/` and `data/phase17/`, committed 2026-09-01 on
  `phase18/setup-integrated-warmstart-g1`; `checkpoints/phase17/` (33.5 GB)
  remains untracked by `.gitignore`, as intended.

---

## 14. Phase 18 — Setup-integrated Phase 8 warmstart  ·  IN PROGRESS

- **Purpose:** produce a fresh Phase 8 C1 warmstart whose policy/value/belief
  learner is integrated with a *beneficial* learned setup policy, correcting the
  five Phase 17 method defects and returning to the Phase 8 supervised
  experimental point rather than self-play.
- **Structure:** an adaptive evidence ladder (gates G0–G6), not a precommitted
  agent sequence. Each stage requires an accepted decision packet before the
  next instruction may be written.
- **Status as of 2026-09-02:** G0 accepted (`P18-D001` PROCEED); G1 closed
  (`P18-D002` REVISE, then `P18-D003` PROCEED: the vs-random margin certified on
  4,096 independent pairs); G2 executed (`P18-D004` REVISE, accepted and
  published at `6afa13be`): implementation parity holds on all 30 method-map
  rows, the setup learner learns a frozen synthetic landscape on all three
  seeds, and the EMA evaluation model lagged severely behind the raw actor
  within the 64-update budget — a predeclared instrument concern. The bounded
  raw-actor confirmation on a fresh landscape (Agent 5, instruction 07)
  delivered `P18-D005` PROCEED for the synthetic trainability portion of G2
  (awaiting review): the raw learner learned in every seed and the median gap
  closure met the frozen 10% threshold at its edge. No setup-only Stratego
  assay, pilot, rehearsal or production run is authorized. See
  `../instructions/phase_18_setup_integrated_warmstart/` and
  `../reports/phase18/`.
