# Project status — canonical source of truth

**Written 2026-08-27. Updated 2026-08-31 by Phase 18 Agent 1** (§13, §14, and the
Phase 17 entries in §9 and §11). This file is the single place that answers "what is
true right now". Where the repository does not support a definitive answer,
this file says **unresolved** rather than guessing.

Superseding rule: if this file and any older document disagree about current
state, **this file wins**. If this file and a *recorded experimental result*
disagree, the recorded result wins and this file is wrong — report it.

Companion documents: [`PHASE_HISTORY.md`](PHASE_HISTORY.md) (what happened),
[`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) (what each artifact is worth),
[`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md) (rules for the next
buildout).

---

## 1. What does "engineering result" mean in this project?

An **engineering result** is a measurement taken to make a build decision. It
has all of these properties:

- it runs on a **compact internal pack** (typically 60–120 paired boards, or a
  few hundred games) against **fixed bots that cannot adapt**;
- its decision rule is **predeclared** before the numbers are seen;
- it reports standard errors but makes **no significance claim**;
- it is valid **only on its named pack** — cross-pack comparison is forbidden
  in conclusions;
- it selects a configuration; it does **not** demonstrate that the selected
  configuration is stronger.

Every Phase 15 and Phase 16 report states this in its own header
(`scientific_validation_status: not performed`). A result becomes something
more only by passing a **separately designed, frozen validation protocol**.
None has been designed since Phase 11, and the Phase 11 sealed test bank is
spent.

Contrast with **`ACCEPTED`**: a result that passed a predeclared formal gate
set under a frozen contract with a sealed or fixed evaluation. Phases 2–10 end
in accepted results. Phases 11B, 12, 15 and 16 do not.

---

## 2. What is the latest accepted direct policy?

```text
checkpoints/phase9/selfplay_c1_v1.pt
sha256        dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model state   f1df694d…
lineage       behavior_B041.pt (post-iteration-40 snapshot), copied byte-identically
parameters    863,959
```

**Status: `ACCEPTED`.** Phase 9 Agent 8's sealed final evaluation passed all 8
hard gates — EWR **0.8442** [0.8228, 0.8647] against the Phase 8 anchor over
1,024 games. Evidence:
[`reports/phase_9_implementation_report.md`](../reports/phase_9_implementation_report.md)
§8 (see lines ~2314 and ~2530). Formally closed at commit `427b963`.

**Nothing has replaced it.** Every later policy candidate (P18, P24, the
Phase 16 training arms) is engineering-grade or unfinished. In particular:

- **Phase 10** delivered an accepted learned *setup selector*, not a policy.
- **Phase 10B** (optional setup-conditioned fine-tuning) is `INCOMPLETE`:
  paused by the operator after 5 of 30 iterations, no classification.
- **Phase 14** never selected a final policy (§5).
- **Phase 17 (tandem self-play)** ran to completion and **promoted nothing**:
  all 24 trained candidates scored *below* the hour-0 start (§13).

A read-only convenience copy exists at
`checkpoints/phase12/phase9_c1_readonly_copy.pt` with **different bytes**
(`81906f71…`). The Phase 9 original is the only binding target.

---

## 3. What are P18 and P24?

They are **intermediate candidate checkpoints from the interrupted Phase 14
run** — the policy/value models at elapsed hour 18 and hour 24. They are not
accepted, not selected, and not final.

| | P18 | P24 |
|---|---|---|
| Phase 14 candidate | hour 18 | hour 24 |
| archive snapshot | `…/stratego_phase14/archive/archive_0009.pt` | `…/stratego_phase14/archive/archive_0012.pt` |
| snapshot sha256 | `c86d8c384daaa23c…` | `3c393d0c8f8b2334…` |
| model-state digest | `9360f2add3dba311…` | `622d9e6caa723c93…` |
| optimizer step | 92,718 | 121,156 |
| iteration | 64 | 82 |
| elapsed | 18.15 h | 24.18 h |
| frozen 128-game pack, mean EWR | **0.8438** (min stratum 0.7656) | **0.7891** (min stratum 0.6562) |
| read-only copy in-repo | `checkpoints/phase15/p18_source_readonly.pt` | `checkpoints/phase15/p24_source_readonly.pt` |

Pack for both: `phase14_checkpoint_selection_pack_v1`, content digest
`896a753b3d568902…`, 128 games across 4 strata. Source: the Phase 14 candidate
ledger at `/Volumes/Brandon_Washington/stratego_phase14/evaluations/phase14_candidate_ledger.json`;
resolution recorded in
[`reports/phase15/agent_01_report.md`](../reports/phase15/agent_01_report.md) §3.

**Two things about P18/P24 that are commonly misread:**

1. **A larger, later re-evaluation reordered them.** An out-of-repo sidecar
   evaluation replayed all ten Phase 14 candidates on 2,200 identical games
   each (`/Volumes/Brandon_Washington/stratego_phase14_sidecar_eval/ANALYSIS.md`).
   At that scale h=18 scored **0.8086** and h=24 scored **0.7886**; h=18 ranked
   first, but **indistinguishable** from h=6 (0.8014), h=12 (0.7909) and h=54
   (0.7905). The sidecar is explicitly *not* a Phase 14 artifact, was fed to no
   ledger, and changed no frozen quantity.
2. **Phases 15 and 16 built on P24, not on the top-ranked h=18.** The Phase 15
   brief named hours 18 and 24 as the two backbones to carry forward; Phase 15
   Agent 2's system selection then chose the P24 pairing on its own pack, and
   Phase 16 inherited P24 as its starting weights. Whether P24 rather than P18
   was the right carry-forward is **unresolved** — no experiment in the
   repository compares the two as *search backbones* at a resolvable margin.

Neither file was ever modified: Phase 15 verified both digests unchanged
before and after its fine-tuning runs.

---

## 4. What are B18 and B24?

**Belief-only specialists** trained in Phase 15 on a fresh, orientation-correct
hidden-piece corpus. They contain no policy and no value tensor — a copied
final C1 block, the encoder norm, a fresh belief MLP (`128 → 512 → 512 → 12`,
GELU), a calibration temperature and identity bindings. Loading one requires
the backbone whose model-state digest it records and refuses any other.

```text
B18  b03685d3557d7c93…   attached to P18   checkpoints/phase15/b18_belief_v1.pt
B24  ac5e15b87f5c5cfd…   attached to P24   checkpoints/phase15/b24_belief_v1.pt
corpus  phase15_belief_corpus_v1  b0493a08d2fcb1dd…
        155,027 observer positions / 4,373,492 supervised hidden pieces
```

Development metrics, all on the same 20,013 positions / 563,558 hidden pieces,
against the accepted `remaining_count_belief_v1` denominator:

| model | CE | R_CE | R_CE 95% CI | top-1 | Brier | ECE |
|---|---|---|---|---|---|---|
| B18 | 1.9745 | 0.9189 | [0.9157, 0.9219] | 0.2930 | 0.8086 | 0.0031 |
| B24 | 1.9709 | 0.9172 | [0.9141, 0.9202] | 0.2944 | 0.8077 | 0.0048 |
| Agent 1C (Phase 11B) | 2.1479 | 0.9996 | [0.9957, 1.0030] | 0.2411 | 0.8468 | 0.0457 |
| count baseline | 2.1488 | 1.0000 | — | 0.2190 | 0.8612 | — |

Calibration was fitted for both and **correctly rejected** — both were already
calibrated, applied temperature 1.0000.

**Status: `ENGINEERING`.** Two limits stated by the phase itself:

- On this corpus the surviving Phase 11B model (Agent 1C) is **statistically
  indistinguishable from the count baseline it was built to beat**. But the
  corpus, the backbone *and* the distribution all changed at once, so this is
  **not** a clean measurement of the orientation defect's cost.
- In match play the learned belief **never separated from a `remaining_count`
  control**: for P18 the count control (0.8667) beat both specialists, for P24
  only B24 (0.8750) beat it (0.7917), and `p18_remaining_count` — which uses
  no learned belief at all — landed within 0.01 EWR of the whole selection.

Evidence: [`reports/phase15/agent_01_report.md`](../reports/phase15/agent_01_report.md),
[`reports/phase15/agent_02_report.md`](../reports/phase15/agent_02_report.md) §5,
handoff [`reports/phase15/phase15_search_handoff_v1.json`](../reports/phase15/phase15_search_handoff_v1.json).

---

## 5. What happened to Phase 14?

**Status: `INTERRUPTED`. Do not describe this run as completed.**

The planned 168-hour final training run began, and was stopped by an operator
emergency stop after **59.97 hours**, at **step 202,504 / iteration 102**, on
**2026-08-24T04:19Z**. The stop request is recorded at
`/Volumes/Brandon_Washington/stratego_phase14/phase14_emergency_stop.json`:

> "operator decision 2026-08-24: stop at ~57h to spend remaining GPU on
> 2,200-game evaluation of all candidates; run preserved and resumable against
> the original window until 2026-08-28T16:15:34Z"

Consequences, precisely:

- The run produced **10 of a planned 29 candidates** — hours 0, 6, 12, 18, 24,
  30, 36, 42, 48, 54. Hours 60 through 168 do not exist.
- `progress.closed` is still `false` and `closed_reason` is `"emergency stop"`.
  The run state remains formally open.
- The frozen selection rule (`phase14_checkpoint_selection_rule_v1`) has a
  `procedure_at_hour_168` that begins "training stops at the frozen deadline
  semantics". **That procedure was never executed. No checkpoint was selected
  under the original completed-run contract, and no final direct policy was
  deployed by Phase 14.**
- The operator has decided the run will **not be resumed** (Phase 16 overview
  §2). Formal closure was assigned to a Phase 16 Agent 5 that has not run, and
  cannot run before the immutable deadline **2026-08-28T16:15:34.689Z** — the
  runner's `finalize` role refuses earlier.
- The repository freeze from that decision still holds: exactly one tracked
  file is modified (`reports/phase13/phase14_launch_manifest_v1.json`, named in
  its own dirty list — a required self-referential rebuild). **Do not commit,
  stash, checkout or clean until the run is formally closed.**

**What the run taught, from its own telemetry:**

- All the measurable learning happened in the first six hours. On the 2,200-game
  sidecar re-evaluation, h=0 → h=6 is **+0.0414** (p < 0.0001); h=6 → h=54
  trends **−0.0342 EWR per 100 h** across 161,000 further optimizer steps.
- Training, not collection, consumed the run: **83% of the 59.97 hours was the
  training phase** (49.57 h) against **17% collection** (10.39 h), and
  minutes-per-iteration grew from 15.7 to 153.8 while collection grew only 4.8
  to 16.3. The cause is **full-game iteration sizing** — 2,048 whole games carry
  more data as games lengthen. Verified read-only in
  [`reports/phase16/agent_03_phase14_decomposition.json`](../reports/phase16/agent_03_phase14_decomposition.json).
- **There is no systematic pack inflation.** The 128-game headline scores are
  noisy, not biased: mean delta to the 2,200-game reading is **+0.0066**. Never
  apply a blanket correction to a headline EWR.

---

## 6. What search systems currently exist?

Three, all built on the same accepted engine `phase12_root_world_search_v1`
(root-sampled worlds, fixed candidate set, greedy rollouts both sides,
`S(a) = Q(a) + beta·log(pi(a) + epsilon)`, `beta = 0.1`, `epsilon = 1e-06`).

| System | Namespace | Models | Status |
|---|---|---|---|
| `phase12_search_player_v1` | `stratego/search/phase12/` | Phase 9 C1 + Agent 1C belief | `CONTAMINATED` for strength claims (§8); the *engine* is accepted and reused unmodified |
| `phase15_search_player_v1` | `stratego/search/phase15/` | P24 + B24 | `ENGINEERING` — Phase 15 selection |
| `phase16_stochastic_search_v1` | `stratego/search/phase16/` | P24 + B24, sampled move + sampled rollouts | `ENGINEERING` — **current candidate** (§7) |

Budgets are shared presets: TINY (8 worlds, depth 4), SMALL (16w/d6),
MEDIUM (32w/d8), plus the closed LARGE (64w/d9) and XLARGE (96w/d11) rungs.
The **oracle** arm — which reads the true hidden army — exists only as an
offline ceiling diagnostic and is refused in production by four independent
refusals.

Playable CLIs: `scripts/play_phase16.py` (current, supersedes
`play_phase15.py`), `scripts/play_phase15.py`, `scripts/play_phase12.py`
(**defective — see §8**).

---

## 7. Which system is the current engineering candidate?

```text
checkpoints/phase16/phase16_stochastic_candidate_v1.json
arm            stoch_t015_r100
move sampling  tau   = 0.15   (softmax over the existing candidate scores)
rollout        tau_r = 1.00, top-p = 0.9
models         P24 622d9e6caa72…  +  B24 ac5e15b87f5c…, belief temperature 1.0
modes          varied_strength = MEDIUM (cap 5.00 s)
               varied_fast     = TINY   (cap 0.91 s)
idle latency   MEDIUM p95 1.760 s / TINY p95 0.252 s
oracle_available_in_production = false
scientific_validation_status   = not performed
```

At `tau = 0, tau_r = 0` it replays the frozen Phase 15 decisions
**bit-identically** — a regression test written before any diagnostic ran.

**This is a selection, not a demonstrated improvement.** On the deciding pack
(`phase16_agent02_interim_pack_v1`, 60 paired boards) the selected arm sits
**+0.0167 ± 0.0640** above the deterministic control at MEDIUM and
**−0.0167 ± 0.0594** below it at TINY. The selection rule picked it for the
**lowest repeat rate among arms not worse than the control**, i.e. for
unpredictability, not for strength. The opponent-side adversarial check
(96 paired boards) gave **+0.0260 ± 0.0394** — recorded as *not a regression*,
not as a gain.

The purpose of the stochastic knob is to resist a human opponent learning the
player's habits across a series. **Fixed bots cannot adapt, so no machine pack
in this repository can measure that property.** The repeat-encounter probe (80
games vs fixed bots) is a weak proxy by construction and is recorded as such.

Nothing has been promoted to "production": the Phase 16 agent charged with
production and the operator exam has not run.

---

## 8. Which search findings reproduced, and which did not?

**Did not reproduce — the central negative result of Phase 16.**

Phase 15 Stage B measured `p24_b24|TINY` at **+0.1375 ± 0.0414** paired against
`p24_direct` on `phase15_match_pack_v1` (120 boards). Phase 16 re-measured the
same arm pair on a fresh pack, `phase16_benchmark_v1` (120 boards drawn from
the same accepted `validation` split by the same machinery, differing only in
board draws and seed streams):

| pack | TINY − direct | MEDIUM − direct |
|---|---|---|
| `phase15_match_pack_v1` | **+0.1375 ± 0.0414** | — (MEDIUM − TINY was +0.0667 on its 60-board ladder) |
| `phase16_benchmark_v1` | **−0.029 ± 0.036** | **−0.017 ± 0.033** |

Absolute baselines on `phase16_benchmark_v1` (120 games each): `p24_direct`
**0.8125**, `p24_b24|TINY` **0.7833**, `p24_b24|MEDIUM` **0.7958**.

Neither Phase 16 gap resolves at 120 boards, and no significance claim is made.
What is plain is that **the +0.10-class search advantage is absent on the second
pack**, and the two packs' paired deltas differ by more than their combined
noise. The finding is that **the search margin is board-draw-sensitive** — this
system's behaviour does not generalize well across setup distributions *even
inside its own accepted library*. No contamination mechanism was found: the
pack ran uncapped, with zero fallbacks in 33,554 decisions.

**Reproduced / held up:**

- **Search beats direct play on the pack it was selected on.** Every Phase 15
  search arm beat its own direct model there, by +0.033 to +0.146 EWR.
- **Deeper search does not help, and the reason is the world distribution.**
  The Phase 15 deep pilot bought 2.19× and 3.90× more compute and got
  **−0.0750 paired at both LARGE and XLARGE**, with the worst opponent falling
  0.833 → 0.667 → 0.500. On the *same* boards and seeds the **oracle improved**
  (+0.042 at LARGE, +0.025 at XLARGE). So the search mechanics scale correctly
  over correct worlds; the sampled-world distribution is what fails. XLARGE is
  also unshippable — idle p95 6.989 s against a 5.0 s ceiling. **The ladder is
  closed.**
- **The oracle ceiling is small.** Perfect hidden-army knowledge is worth only
  **+0.100 EWR (P18)** and **+0.146 (P24)** at TINY, and moves only 10–12% of
  decisions. Belief quality is not the binding constraint.
- **Mixing belief with the count baseline does not rescue it.** The Phase 15
  mixture pilot closed at Stage 1 with no useful mixture; position-level oracle
  Q-regret could not even distinguish `b24@MEDIUM` (EWR 0.9333) from
  `b24@LARGE` (0.8583) — **+0.0008 ± 0.0017**, 110 of 120 positions tied.

**Adversarial setups — measured, and smaller than expected.** On
`phase16_adversarial_baseline_v1` (96 paired boards per arm), swapping the
opponent's army to an adversarial one cost **0.0625 at TINY** (predeclared
reading `between_predeclared_thresholds`) and **0.0469 at MEDIUM**
(`weakens_distribution_hypothesis`). The cost is **not uniform**: it
concentrates in `spy_shadow` (−0.25 TINY / −0.29 MEDIUM) and the bombed-flag
families, while `miner_wall`, `decoy_flag_structure` and `free_novelty` cost
nothing or score above control. 12-pair strata carry SEs of 0.08–0.21.

**Training-recipe shootout — no verdict.** Three 6-hour arms (control vs two
damped-schedule recipes, one with expanded setups) all started from the
saturated P24, and the predeclared rule returned **STOP: no long run is
authorized by this file**. Read it correctly: **it could not tell the three
arms apart**, not that damping is worse. Every arm's whole h-curve sits inside
one standard error of its own starting point; the 0.03 decision margin is
**0.53 SE** of the 60-board instrument, and the same games scored over the full
120-board pack **reverse the verdict**. Arm C trained on adversarial setups and
scored *lower* on them (−0.073) — which likewise establishes nothing, since a
six-hour run from a saturated start cannot separate "does not help" from "needs
a longer horizon".

What the shootout *did* establish is infrastructural and solid: a window-based
collector holds iteration wall-time to a coefficient of variation near 0.05
across ~300 iterations where Phase 14's grew 8× over 102; three arms ran six
hours each with zero vetoes and zero non-finite losses; the window-edge
invariant is exact (0.000000000). **Collection speed itself is a wash (0.90×
against Phase 14's production median); the real win is pinned iteration
sizing.**

---

## 9. Which evidence is superseded or contaminated?

**`CONTAMINATED` — a Blue setup-orientation defect.**
`Phase11BSetupSources.draw` returned canonical own-orientation tuples and the
old glue handed them straight to `create_game()` for Blue. Canonical rank 0 is
a player's *own* back rank and Blue's engine setup order runs front-to-back, so
**an unoriented Blue army was placed reversed**. Quantified in Phase 15 Agent 1
§2 over 4,096 paired boards: the corrected path produces **1.77%** front-row
flags (145 of 8,192 armies); the old glue would have produced **77.00%**
(3,154 of 4,096 Blue armies) — which reproduces Phase 12's 47-of-64 observation
almost exactly.

Invalid for strength claims:

- **`phase11b_common_corpus_v1`** and every belief metric measured on it,
  including Agent 1C's `R_CE 0.9459` and the whole Phase 11B leaderboard.
- **The Phase 12 match packs** and every EWR from them, including
  `Agent4_quick_EWR 0.6406` (TINY) and `0.6875` (MEDIUM).

The correct rule, re-derived from the engine's own `SETUP_SQUARES`:

```text
red engine row == canonical rank ;  blue engine row == 9 - canonical rank
```

**Still live, unfixed:** `scripts/play_phase12.py` (lines ~209–250) draws human
setups through `Phase11BSetupSources` and hands them straight to `create_game`.
It was found and reported in Phase 15 and deliberately **not** repaired under
the repository freeze. `scripts/play_phase15.py` and `scripts/play_phase16.py`
go through the Phase 15 orientation gate and are unaffected.

**`SUPERSEDED` (correct when measured, replaced by later work):**

- Phase 11B belief models → superseded by B18/B24 on the corrected corpus.
- Phase 12's `phase12_search_candidate_v1` → superseded as a *candidate* by
  `phase15_search_candidate_v1`, then by `phase16_stochastic_candidate_v1`.
  The search *engine* it names is still the accepted engine everything runs on.
- Phase 15's search-advantage reading → not superseded but **not reproduced**
  on `phase16_benchmark_v1` (§8). Both readings stand; neither is deleted.
- The **85% effective-win-rate target vs casual humans** → formally retired
  2026-08-25 for lack of a measurable human pool, replaced by `phase16_goal_v1`.
  **It was retired, not achieved.**

**`FAILED GATE` (stands as a result):** Phase 11's sealed test, `R_CE 0.9746`
[0.9726, 0.9764] against Gate A's `<= 0.97`, over 2,048 paired cases / 4,096
games. Gates B–H passed. Classification **FAIL**; Phase 12 was *not* authorized
by Phase 11. The sealed test bank is **permanently spent** — a future belief
repair phase needs fresh sealed evidence.

---

## 10. What evaluation remains pending?

| Item | Status |
|---|---|
| Operator re-baseline series (10 games) | **`PENDING`** — protocol and logging delivered, **zero games played** |
| Operator exam (20 games, pass = model EWR ≥ 0.50) | **`PENDING`** — never run |
| `operator_harvest` adversarial family | **`PENDING`** — present but **empty**, 0 setups |
| Joint/autoregressive belief world model + deep-ladder rerun | **`PENDING`** — specified (Phase 16 Agent 4), **never built**; no `stratego/belief/phase16/` exists |
| Formal Phase 14 closure | **`PENDING`** — blocked until 2026-08-28T16:15:34Z |
| Any human-strength claim | **`PENDING`** — see below |

`data/phase16/operator_games.jsonl` holds exactly **one** line, and it is the
CLI self-play verification game (`varied_fast` vs `varied_strength`, 169 plies),
**not** an operator game.

**No human-strength evidence exists in this repository.** The 85% target was
never measured and is retired. Its replacement, `phase16_goal_v1`, has never
been run. All previous informal human impressions were retired by Phase 16.
Any future human claim requires a separately frozen human-evaluation protocol
(see [`EXPERIMENT_FRAMEWORK.md`](EXPERIMENT_FRAMEWORK.md) §5).

---

## 11. Is a new long training run currently authorized?

**No.**

- The Phase 16 recipe shootout's predeclared `stop_rule` fired: *"if neither B
  nor C clears adopt_recipe: STOP, write the report, hand back to the operator;
  **no long run is authorized**"*. Recorded in
  [`checkpoints/phase16/phase16_recipe_candidate_v1.json`](../checkpoints/phase16/phase16_recipe_candidate_v1.json)
  (`adopt_recipe.pass = False`).
- The Phase 14 run is not resumable in practice (operator decision: no resume)
  and not formally closed.
- The agent charged with authorizing and running production work has not run,
  and is gated on the 2026-08-28T16:15:34Z deadline plus prerequisites that are
  themselves incomplete.

A new buildout is **anticipated** — incorporating further ideas from Ataraxos
and restarting training from the Phase 14 stage — but as of this file's date it
is **not designed, not specified and not authorized**. Nothing in this
documentation set constitutes authorization to design or start it.

---

## 12. Known unresolved questions

These are genuinely open; the repository does not answer them.

1. **P18 vs P24 as the carry-forward backbone.** No experiment compares them as
   search backbones at a resolvable margin. Phase 15's Stage A oracle-agreement
   and Stage B match results point in different directions and the pack cannot
   separate them.
2. **Whether a learned belief head is needed at all.** `p18_remaining_count`
   scores within 0.01 EWR of the whole Phase 15 selection while using no
   learned belief. The specialists' only distinct signal is decision-level
   oracle agreement, which did not convert into a match-level separation.
3. **Why the search margin flipped between two packs from the same split.**
   Localized (TINY loses ground vs p18 −0.125, anchor −0.167, miner_rush
   −0.167) but not explained.
4. **Whether expanded/adversarial training setups help.** Arm C moved its
   training distribution hard (mean game length 855.6 plies vs 592.7) and its
   strength did not move. Six hours from a saturated start cannot separate "does
   not help" from "needs a longer horizon".
5. **Whether the stochastic knob actually buys adaptation resistance.**
   Unmeasurable against fixed bots by construction. Only the operator exam can
   answer it.
6. **Where the operator's wins actually come from.** Adverse setups alone cost
   only 0.047–0.063 EWR against the machine roster — less than the hypothesis
   predicted. Either the operator's setups are outside this library, or the
   mechanism is predictability across a series, which no single-game pack
   measures. **Unresolved.**

---

## 13. What happened in Phase 17 (tandem self-play)?

**Note on the number.** "Phase 17" is ambiguous. The *planned* Phase 17 was
casual human evaluation and is still `PENDING` (§10). The *executed* Phase 17 is
tandem current-policy self-play, `RUN-2026-B`. This section is about the latter.
See [`PHASE_HISTORY.md`](PHASE_HISTORY.md) §12 and §13.

**Status: `COMPLETE`. Result: negative. No checkpoint promoted.**

```text
run                RUN-2026-B, launched from 90278aa
duration           12.658 active hours, 535 of a frozen 640 iterations
termination        operator, after the twelfth hour
candidates         25 paired (move + setup) EMA exports, all byte-verified
integrity events   0 in all 535 telemetry rows
evaluation         2026-08-30, 120-board phase17_composite_benchmark_v1,
                   both lanes, 0 refusals, bit-deterministic across worker counts
```

The four headline readings:

| Question | Result |
|---|---|
| Did move-only improve over hours 6–12? | **No — it degraded.** Slope −0.0115 EWR/h, t = −2.97 |
| Did the joint lane improve? | **No — flat.** Slope +0.0003 EWR/h, t = 0.04 |
| Did any trained candidate beat the hour-0 start? | **No — 0 of 24.** The move-only curve peaks at hour 0 |
| Did the learned setup policy beat the fixed library? | **No — −0.0679 EWR, t = −2.91.** It also never beat its own random initialization (DiD +0.0237, t = +0.44) |

**What this does and does not establish.** It is a valid negative result *for the
exact implementation that was run*. It is **not** evidence that the paper's setup
method fails on this project, for two independent reasons:

1. **The implementation differed from the authors'.** Phase 17's method map was
   built from the paper alone. The authors' code is now available and differs
   materially — entropy units in the advantage, forced flag handedness plus
   post-generation reflection, reusable setup pools with averaged outcomes, and a
   1,024-episode effective batch against Phase 17's 32. Row-by-row in
   [`../reports/phase18/ataraxos_setup_method_map_v2.md`](../reports/phase18/ataraxos_setup_method_map_v2.md).
2. **The evaluation could not have resolved the effect.** The 120-board lane has a
   **minimum detectable effect of 0.138 EWR** at 80% power, computed from Phase
   17's own per-case paired outcomes. Every reading above except the setup-vs-library
   comparison sits inside that band. This also reproduces the independently measured
   0.1435 EWR pure-noise spread for 25 candidates on a 120-game lane.

**Evidence.** `reports/phase17/agent_05_report.md`, `agent_07_report.md`,
`phase17_run_closeout_v1.json`, `local_eval/`. All Phase 17 evidence is
preserved unmodified. **Committed 2026-09-01** by Phase 18 Agent 2 on
`phase18/setup-integrated-warmstart-g1`, byte-for-byte as its producing agents
wrote it, because Gate G1 must bind an immutable source closure.
`checkpoints/phase17/` holds 33.5 GB of unpruned run checkpoints and remains
untracked by `.gitignore`, as intended.

**A known evaluator defect, found and deliberately not fixed.** In
`stratego/evaluation/phase17/evaluator.py`, a refusal receipt is written to
`<candidate_id>.result.json`; `existing_result()` then finds that file on every
later attempt and refuses the candidate as duplicate-conflicting — permanently.
A candidate that failed transiently cannot be re-evaluated without deleting the
file by hand. It did not affect the Phase 17 batch. Any future evaluator must
carry this as a regression case.

---

## 14. Is Phase 18 authorized?

**Gate G1 is CLOSED (P18-D003 accepted 2026-09-02). Gate G2: P18-D004 = `REVISE`,
accepted 2026-09-02 and published at `6afa13be` — parity passes and the learner
learns the synthetic landscape; the EMA-based criterion was not met within the
64-update budget because the EMA (0.999^64 = 0.937975 of the initial parameter
contribution retained, ~1,000-update time constant) lagged severely behind the raw
actor. Agent 5's bounded raw-actor confirmation on a fresh landscape is
authorized and IN PROGRESS. Nothing beyond G2 is authorized.** *(Updated
2026-09-02 by Phase 18 Agent 5.)*

Phase 18 — *setup-integrated Phase 8 warmstart* — was planned on 2026-08-31. Its
goal is a fresh Phase 8 C1 warmstart whose policy/value/belief learner is
integrated with a **beneficial** learned setup policy, correcting the Phase 17
method defects and returning to the Phase 8 supervised experimental point instead
of self-play.

```text
executed   01_AGENT_1 (G0, 2026-08-31)          -> P18-D001 PROCEED (accepted)
executed   04_AGENT_2 (G1 control, 2026-09-01)  -> P18-D002 REVISE  (accepted; 42/42
           gates, 7/8 margins; vs-random uncertifiable at 1,024 pairs)
executed   05_AGENT_3 (G1 confirmation, 2026-09-02) -> P18-D003 PROCEED (ACCEPTED
           2026-09-02; delta +0.006348, 95% [+0.000793, +0.011902] on 4,096
           independent pairs vs the -0.010 margin -> G1 CLOSED; branch
           phase18/g1-random-confirmation published at ef7523c1, local == remote)
executed   06_AGENT_4 (G2 setup parity + synthetic assay, 2026-09-02) -> P18-D004 REVISE
           (ACCEPTED 2026-09-02; parity 30/30 + oracle PASS, zero integrity events;
           raw actor closes 20.9/18.5/14.8% of the gap; the EMA, 0.999^64 = 0.937975
           retained (~1,000-update time constant), lagged severely behind it and
           closes a median 0.35% vs 10% -> predeclared instrument concern; branch
           phase18/g2-setup-parity PUBLISHED at 6afa13be, local == remote)
in progress 07_AGENT_5 (G2 bounded raw-actor confirmation, 2026-09-02) -> P18-D005
           pending (fresh landscape and seeds, unchanged G2 method, raw actor primary
           for this synthetic assay only, EMA telemetry secondary; branch
           phase18/g2-raw-confirmation, NOT pushed)
NOT authorized  the setup-only Stratego assay (G3), the tandem pilot (G4), the
                production rehearsal (G5), and the full run (G6); no Stratego setup
                training and no sealed Phase 8 access inside G2
```

Phase 18 is an **adaptive evidence ladder** (gates G0–G6), not a precommitted
agent sequence. Every stage stops at a decision packet that the operator and the
reviewing chat must accept before the next instruction may be written.

Two rule-identity facts recorded on 2026-09-02 (see
`reports/phase18/phase18_rule_identity_errata_v1.json`): the `P18-D003.json`
narrative names a battleless move limit of 100 where the frozen contract, the engine
constant, the schedule and all 16,384 receipts carry the accepted **evaluation** value
200 — a metadata error, the packet is not rewritten and G1 is not rerun; and the
frozen `phase18_evaluation_contract_v1.json` names *training* rules (100) for the
future play lanes, so the training-versus-evaluation rule choice **must be amended
explicitly before any real-game G3/G4 evaluation**.

Gate G2's open item (P18-D004): the synthetic assay's decision read the EMA model,
which at the paper's 0.999 smoothing updated once per setup update retained
0.999^64 = 0.937975 of its initial parameter contribution after the 64-update
budget (an approximately 1,000-update time constant) and lagged severely behind
the raw actor in the frozen assay (1.3–2.8% of the raw displacement). The accepted
bounded correction (Agent 5, instruction 07) is an independent confirmation on a
fresh landscape with fresh seeds in which the synthetic assay's decision reads the
raw generation actor; the EMA remains the required evaluation/deployment model for
every later Stratego-facing stage. A pass closes only the synthetic trainability
portion of G2 and does not authorize G3 or the full warmstart.

Agent 1's outputs are in `reports/phase18/`. Two dependencies are recorded as
blocking later gates and neither blocks the Phase 8 control:

- the `unusual_procedural` setup pack **does not exist and cannot be built from
  existing assets** — the entire 8,000-board setup library is already consumed by
  the accepted Phase 8 corpus (blocks G4); and
- the `operator_sealed` pack requires operator-supplied setups (blocks G6).
