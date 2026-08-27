# Phase 16 — Agent 1
## Measurement: backup, canonical benchmark, adversarial setups, operator protocol

This is an engineering deliverable, not a scientific claim.
`scientific_validation_status: not performed`. No significance claim is made
anywhere in this report; every table carries its own game or position count,
every EWR names its pack, and cross-pack comparisons are drawn nowhere.

**What was delivered:** the T0/end-of-task backup record; `phase16_benchmark_v1`
(120 frozen paired boards + a resumable scoring runner with a provider plug-in
interface and a predeclared 60-board quick subset); `phase16_adversarial_setups_v1`
(9 families, 96 gate-validated setups, `operator_harvest` present-but-empty);
the 3-arm paired baseline measurement at TINY and MEDIUM (576 games total);
and the operator series protocol with logging, capture and harvest tooling.

**Headline baselines** (`phase16_benchmark_v1`, 120 games each): `p24_direct`
**0.8125**, `p24_b24|TINY` **0.7833**, `p24_b24|MEDIUM` **0.7958**.

**Two findings that belong up front:**

1. **The Phase 15 search advantage does not appear on this pack.** Paired on
   the same 120 boards, TINY − direct = −0.029 ± 0.036 and MEDIUM − direct =
   −0.017 ± 0.033, against Phase 15's +0.1375 ± 0.0414 on its own boards.
   Neither gap here is resolvable at 120 boards, but the two packs' paired
   deltas differ by more than their combined noise: the search margin is
   board-draw-sensitive (section 4).
2. **Adverse setups alone cost little overall — and the cost is concentrated.**
   Arm 2 − arm 1 drop: 0.0625 at TINY (`between_predeclared_thresholds`),
   0.0469 at MEDIUM (`weakens_distribution_hypothesis`). The damage sits in
   `spy_shadow` and the bombed-flag families at both presets; several families
   cost nothing (section 5). The per-family table is Agent 3's input.

The operator re-baseline is pending (operator unavailable this window); the
instruments are ready.

## 1. Process boundary (brief section 1)

Checked 2026-08-25T20:31Z, recorded in `reports/phase16/agent_01_process_boundary.json`:

- Phase 14 run state (read-only, from
  `/Volumes/Brandon_Washington/stratego_phase14/phase14_run_state.json`):
  `closed: false`, stopped at 59.97 h / step 202,504 / iteration 102, deadline
  `2026-08-28T16:15:34.689Z`. Nothing was signalled, edited, rotated or
  finalized.
- The only live process matching the stack: the accepted read-only Phase 14
  dashboard on port 8714 (pid 99915). No learner, supervisor or collector is
  running.
- The repository freeze holds: the sole tracked diff is
  `reports/phase13/phase14_launch_manifest_v1.json`, exactly as required, and
  every Phase 16 Agent 1 artifact is additive and untracked.

## 2. Namespaces

```text
stratego/evaluation/phase16/     contract, benchmark, adversarial, baseline,
                                 runner, analysis, operator_log
tests/evaluation/phase16/        88 new tests
scripts/run_phase16_agent01.py   build / bench / baseline / analyse / handoff /
                                 harvest / backup roles, compute-lock aware
scripts/phase16_capture_setup.py operator setup capture (4x10 grid)
scripts/play_phase16_operator.py logged operator play vs the frozen Phase 15 player
data/phase16/                    the three frozen instruments + operator log
reports/phase16/                 this report, JSON artifacts, pack results
```

Nothing accepted was modified. Phase 15's match machinery (`matchplay`,
`systems`, `boards`, `loaders`, `analysis`), the orientation gate and the
accepted setup library are consumed by import only.

## 3. T0 — the untracked backup (brief section 3)

Recorded in `reports/phase16/agent_01_backup.json`.

- **T0 (before any Phase 16 work, run by the coordinator 2026-08-25):**
  `/Volumes/Brandon_Washington/stratego_untracked_backup_20260825.tar` —
  8,137,132,544 bytes, 269 tar entries, sha256
  `712f4c4aaecd6812182db628d2131b444062f2e6a293b391ed0bdd5187088b56`, filelist
  alongside, `*_prefix_*.npy` excluded (regenerable, ~15 GB). Verified present
  and size/digest-consistent this session; per the coordinator's direction it
  was **not** redone.
- **End-of-task refresh (2026-08-26T05:39:56Z, after every Agent 1 artifact
  existed):**
  `/Volumes/Brandon_Washington/stratego_untracked_backup_20260826.tar` —
  8,145,454,592 bytes, 381 tar entries from 25 untracked roots, sha256
  `18fd6e3961cb82a76cd143f7e1f1e697c23517ea8375f955a1b02cc33e75513d`,
  filelist alongside, same exclusion. Neither backup deletes the other. (The
  archived copy of this report predates this paragraph by minutes — the
  backup cannot contain its own record; everything else, including all pack
  results, manifests and the handoff, is inside.)

## 4. `phase16_benchmark_v1` — the canonical pack (brief section 4)

Frozen at `data/phase16/phase16_benchmark_v1.json`;
manifest digest `ebd130198ea500248b32df990bee876583a10d53546f38a6346ec522407320c2`.

- **120 paired boards** = {10 opponents} x {3 setup sources} x {2 colours} x 2
  ordinals. Balance by construction: 12 per opponent, 40 per source, 60 per
  colour. Opponent roster = Phase 15 Stage B (p18, p24, phase9_anchor, the two
  rule-based, the five stress styles), re-exported from the Phase 15 contract
  rather than restated.
- Boards drawn fresh from the accepted library's `validation` split through
  the imported Phase 15 setup sources under **Phase 16 seeds** (blake2b
  personalization `strat-p16m`, namespace `phase16.agent1`); no Phase 15 or
  Phase 12 board is reused. Every board passed the imported section-4
  orientation gate before `create_game`.
- The manifest is executable: `materialize_benchmark` rebuilds every board
  from its id and refuses the manifest on any byte difference (verified at
  build time and pinned by tests).
- **Quick subset** (`quick60`, predeclared): ordinal 0 of every cell — 60
  boards, balanced by construction, for training-run checkpoint scoring.
- **Scoring runner**: `stratego.evaluation.phase16.runner.score_on_benchmark
  (mode_or_provider, preset, workers, subset=None)` scores (a) any Phase 15
  production pairing id and (b) any object implementing the Phase 15
  decision-seat interface via a picklable factory spec
  (`{"factory": "module:callable", "kwargs": {...}, "arm_id": ...}`) — the
  plug-in path for Agents 2/3/4. The oracle is refused by name. Packs resume
  from their JSONL result file: a killed run re-runs only missing games.

### Baselines on the full pack

All EWRs in this section are on `phase16_benchmark_v1` (120 games each, 10
workers, uncapped; machine-readable strata with counts in
`reports/phase16/agent_01_benchmark_baselines.json`).

| baseline | games | W/D/L | EWR | worst opponent (n=12) | move change | fallbacks | minutes (10w) |
|---|---|---|---|---|---|---|---|
| p24_direct\|direct | 120 | 96/3/21 | 0.8125 | 0.500 (p24) | - | 0 | 0.2 |
| p24_b24\|TINY | 120 | 92/4/24 | 0.7833 | 0.542 (p18) | 0.064 | 0 | 65.6 |
| p24_b24\|MEDIUM | 120 | 94/3/23 | 0.7958 | 0.500 (p24) | 0.058 | 0 | 162.5 |

Per-opponent EWR (n=12 each):

| baseline | p18 | p24 | anchor | strat | tact | scout | miner | bersk | miser | chaos |
|---|---|---|---|---|---|---|---|---|---|---|
| p24_direct\|direct | 0.667 | 0.500 | 0.750 | 0.750 | 0.875 | 0.667 | 0.917 | 1.000 | 1.000 | 1.000 |
| p24_b24\|TINY | 0.542 | 0.542 | 0.583 | 0.750 | 0.917 | 0.750 | 0.750 | 1.000 | 1.000 | 1.000 |
| p24_b24\|MEDIUM | 0.750 | 0.500 | 0.583 | 0.583 | 0.917 | 0.792 | 0.917 | 0.917 | 1.000 | 1.000 |

Per setup source (n=40), direct / TINY / MEDIUM: neutral_v1
0.825 / 0.725 / 0.763, phase14_learned 0.825 / 0.838 / 0.825,
targeted_family 0.788 / 0.788 / 0.800. Per colour (n=60): red
0.833 / 0.808 / 0.858, blue 0.792 / 0.758 / 0.733. (Full strata with counts
in the JSON.)

### Finding: on this pack, TINY search scores BELOW direct

`p24_b24|TINY` 0.7833 vs `p24_direct` 0.8125 — **paired on the same 120
boards: TINY − direct = −0.029 ± 0.036 SE** (9 boards better / 99 identical
outcome / 12 worse). Phase 15's Stage B measured the same arm pair at
**+0.1375 ± 0.0414 paired** on its own 120-board pack
(`phase15_match_pack_v1`). Three things about the reading:

1. **On this pack the difference is inside one standard error of zero** — a
   120-board engineering pack cannot resolve a 0.03 gap, and no significance
   claim is made. What *is* plain is that the +0.10-class search advantage
   Phase 15 saw is **absent here**.
2. **The two packs disagree by more than their combined noise.** The paired
   deltas differ by ~0.17 with SEs of ~0.04 each; treating both as draws
   around one common true value is not tenable at these SEs. The packs
   differ only in board draws and seed streams (`strat-p15s` vs
   `strat-p16m`), drawn from the same accepted `validation` split by the
   same machinery — so the search gain is **board-draw-sensitive**, which is
   itself evidence for the phase's premise that this system's behaviour is
   distribution-sensitive. The per-opponent split localises the reversal:
   TINY loses ground against the neural opponents (p18 −0.125, anchor
   −0.167) and stress_miner_rush (−0.167), while gaining slightly against
   p24 (+0.042) and stress_scout_rush (+0.083).
3. **No contamination mechanism was found**: the pack ran uncapped, zero
   fallbacks in 33,554 player decisions, max contended move 0.78 s (idle
   equivalent well under the deployed 0.91 s cap), and the direct arm's rows
   were played by the same accepted seats on the same boards and seeds.

**MEDIUM shows the same picture.** Paired on the same 120 boards:
MEDIUM − direct = **−0.017 ± 0.033** (9 better / 101 identical / 10 worse)
and MEDIUM − TINY = +0.013 ± 0.030 (9/105/6). Phase 15's Stage C ladder had
measured MEDIUM − TINY at +0.0667 paired on its 60-board pack. On
`phase16_benchmark_v1`, neither search preset separates from direct play in
either direction beyond one standard error — the search advantage simply
does not appear on this board draw. (MEDIUM pack: zero fallbacks in 29,424
decisions, max contended move 3.68 s, uncapped.)

Consequence for Phase 16: `phase16_benchmark_v1` is the canonical instrument
going forward, and Agents 2/3 should read *paired deltas on this pack* rather
than assuming Phase 15's search margin transfers. The margin's sensitivity
to the board draw is itself the most Phase-16-relevant fact this pack
produced: it is exactly the behaviour the deep-search pilot saw from the
other side (belief-sampled worlds anti-scaling), and it says strength claims
for this system generalise poorly across setup distributions even inside the
accepted library.

## 5. `phase16_adversarial_setups_v1` — the pack that models the operator (brief section 5)

Frozen at `data/phase16/phase16_adversarial_setups_v1.json`:
library digest `e01529cedc042e858bdf9f9036e5da48c552077dd7a5a6a1eef411393afa58e5`,
authored digest `dcafa1614d3a5be9867bebf7a975b984243eca9bb331f3be42cc8c571067deb3`
(the authored digest covers the eight authored families and is frozen; a
harvest append bumps `harvest_revision` and the library digest only).

- **9 families, 96 setups** (8 authored x 12 + `operator_harvest`, present but
  empty — see section 6). Stored in the accepted setup-library representation:
  canonical own-orientation 40-tuples plus family metadata. Every entry passed
  the imported Phase 15 section-4 gate (flag row, legal rows, exact inventory,
  paired-mirror on the oriented output), plus a per-family structural
  signature check pinned by tests.
- Authored from documented human conventions and the Ataraxos setup analysis;
  deterministic from `derive_measure_seed('measure_adversarial_author',
  family, ordinal)` and internally varied (corner side, shell shape, lanes,
  spreads and fills all rotate with the ordinal).

| family | setups | signature (checked) |
|---|---|---|
| operator_harvest | 0 | operator-captured only; empty with TODO |
| bombed_corner_flag | 12 | flag a-/j-corner rank 0, every orthogonal approach bombed |
| bombed_center_flag | 12 | flag rank 0 files 3-6, complete 3-bomb shell |
| scout_screen | 12 | >= 6 scouts on the front rank, high pieces ranks 1-2 |
| aggressive_marshal | 12 | marshal on the front rank |
| spy_shadow | 12 | spy at rank >= 2 within distance 2 of the general; colonel+bomb trap |
| miner_wall | 12 | all 5 miners on rank 2, file spread >= 6 |
| decoy_flag_structure | 12 | bomb-ringed decoy corner holding a low piece; true flag unbombed |
| free_novelty | 12 | convention breakers: mid-board flags, front bombs, wing-massed bombs |

### The baseline measurement (`phase16_adversarial_baseline_v1`)

Frozen at `data/phase16/phase16_adversarial_baseline_v1.json`, manifest digest
`e937df0dc395462ffce9090964009b139a3fca71060814d8bddce794d5219f86`, binding
the library digest above. **96 paired board triples** (one per authored setup), player =
`p24_b24` (the Phase 15 selection), at TINY and MEDIUM:

```text
arm 1  benchmark_control     opponent army from the accepted library (validation)
arm 2  adversarial_opponent  opponent army = the pair's adversarial setup
arm 3  adversarial_both      both armies from the pack (same family,
                             entry offset +6 for the player; secondary)
```

The arms of a pair share the machine opponent, the player's colour, the match
seed (so every rule-opponent decision stream), and — arms 1-2 — the player's
own setup; only the army the arm is about differs. Per family: 12 pairs, all
10 opponents covered, colours 6/6. Player setups rotate the three accepted
sources (neutral_v1 / phase14_learned / targeted_family) by pair index.

### Results at TINY (288 games: 96 per arm, player p24_b24|TINY, 38.8 min on 10 workers)

| arm | games | W/D/L | EWR |
|---|---|---|---|
| benchmark_control | 96 | 80/1/15 | 0.8385 |
| adversarial_opponent | 96 | 73/3/20 | 0.7760 |
| adversarial_both | 96 | 81/1/14 | 0.8490 |

- **arm 2 − arm 1 (paired, 96 pairs): −0.0625 ± 0.0498 SE** (9 pairs better /
  70 identical / 17 worse) → **drop = 0.0625**.
- arm 3 − arm 1 (paired, 96 pairs): +0.0104 ± 0.0474 — giving the *player*
  adversarial setups too roughly cancels the opponent's gain at TINY.

Per family, arm 2 − arm 1 (12 pairs each):

| family | control EWR | adversarial EWR | delta | SE |
|---|---|---|---|---|
| spy_shadow | 0.8750 | 0.6250 | **−0.2500** | 0.115 |
| bombed_corner_flag | 0.8333 | 0.6250 | **−0.2083** | 0.208 |
| bombed_center_flag | 0.8333 | 0.7500 | −0.0833 | 0.149 |
| scout_screen | 0.9167 | 0.8750 | −0.0417 | 0.042 |
| aggressive_marshal | 0.8333 | 0.8333 | 0.0000 | 0.174 |
| decoy_flag_structure | 0.7500 | 0.7500 | 0.0000 | 0.213 |
| free_novelty | 0.9167 | 0.9167 | 0.0000 | 0.000 |
| miner_wall | 0.7500 | 0.8333 | +0.0833 | 0.083 |

**Predeclared reading at TINY: `between_predeclared_thresholds`** — the
overall drop (0.0625) is neither >= 0.10 (confirm) nor < 0.05 (weaken). Stated
plainly: at TINY the adversarial pack costs the system about six points of
EWR overall, and the cost is concentrated in exactly two families —
`spy_shadow` and `bombed_corner_flag` (the classic human convention) — while
five families barely move it. The per-family table above is Agent 3's
training-mixture input regardless of the overall band, with the caveat that
12-pair strata carry SEs of 0.1-0.2.

### Results at MEDIUM (288 games: 96 per arm, player p24_b24|MEDIUM; arms 1-2 in 225.6 min, arm 3 resumed for a further 68.5 min, 10 workers)

| arm | games | W/D/L | EWR |
|---|---|---|---|
| benchmark_control | 96 | 83/1/12 | 0.8698 |
| adversarial_opponent | 96 | 77/4/15 | 0.8229 |
| adversarial_both | 96 | 79/1/16 | 0.8281 |

- **arm 2 − arm 1 (paired, 96 pairs): −0.0469 ± 0.0432 SE** (8 pairs better /
  75 identical / 13 worse) → **drop = 0.0469**.
- arm 3 − arm 1 (paired, 96 pairs): −0.0417 ± 0.0436 — at MEDIUM, unlike
  TINY, giving the player adversarial setups too does *not* cancel the
  opponent's gain. (Full three-arm pack: 288 games.)

Per family, arm 2 − arm 1 (12 pairs each):

| family | control EWR | adversarial EWR | delta | SE |
|---|---|---|---|---|
| spy_shadow | 0.9167 | 0.6250 | **−0.2917** | 0.130 |
| bombed_center_flag | 1.0000 | 0.7500 | **−0.2500** | 0.115 |
| bombed_corner_flag | 0.8333 | 0.7083 | −0.1250 | 0.125 |
| scout_screen | 0.8333 | 0.7500 | −0.0833 | 0.083 |
| aggressive_marshal | 0.9583 | 0.9167 | −0.0417 | 0.097 |
| free_novelty | 0.8333 | 0.9167 | +0.0833 | 0.149 |
| decoy_flag_structure | 0.8333 | 1.0000 | +0.1667 | 0.112 |
| miner_wall | 0.7500 | 0.9167 | +0.1667 | 0.112 |

**Predeclared reading at MEDIUM: `weakens_distribution_hypothesis`** — the
overall drop (0.0469) falls just under the 0.05 line. Stated plainly, both
ways, as the brief requires:

- The **overall** adversarial cost is small at both presets: 0.0625 (TINY,
  between thresholds) and 0.0469 (MEDIUM, formally "weakens"). Adverse
  *setups by themselves* do not reproduce a >= 0.10 collapse against the
  machine roster. If the operator's wins come mostly from setups, they come
  from setups this library does not capture — or from the second mechanism
  (predictability across a series), which no single-game pack can measure.
- The cost is **not uniform**: it concentrates, consistently at both
  presets, in `spy_shadow` (−0.25 TINY / −0.29 MEDIUM) and the bombed-flag
  families (corner −0.21 / −0.13; center −0.08 / −0.25), while
  `miner_wall`, `decoy_flag_structure` and `free_novelty` cost nothing or
  even score *above* control. The per-family table is Agent 3's
  training-mixture input; 12-pair strata carry SEs of 0.08-0.21.

## 6. Operator series protocol and logging (brief section 6)

Delivered:

- `reports/phase16/operator_protocol_v1.md` — the re-baseline series (10
  games) and the exam (20 games, pass = model EWR >= 0.50), conditions (idle
  machine, alternating colours, no operator time pressure, free adaptation),
  and the EWR-by-game-index trend reading.
- `stratego/evaluation/phase16/operator_log.py` — one JSON line per game,
  schema `phase16_operator_game_v1` (timestamp, script+mode, seats, colours,
  both setups as canonical tuples + family id when drawn, full action history,
  result, ply count, per-move wall times for both seats);
  `operator_series_summary` reports machine EWR by game index with a running
  mean; `harvest_operator_setups` extracts operator setups into
  `operator_harvest` (dedup by tuple, gate-validated).
- `scripts/play_phase16_operator.py` — the Phase 15 player (imported,
  digest-checked, unedited) wrapped with logging; supports operator-entered
  setups in the capture-grid format. Stands in until Agent 2's
  `play_phase16.py`.
- `scripts/phase16_capture_setup.py` — 4x10 grid capture into
  `operator_harvest`, with exact-inventory diagnostics and a dry-run mode.
- CLI harvest: `run_phase16_agent01.py --role harvest`.

**Status: the operator was not available during this run.** The re-baseline
series is **pending** (instrument delivered, no games played);
`operator_harvest` is present but **empty** — TODO for the first operator
session: play the 10 re-baseline games, then run `--role harvest`.

## 7. Handoff (brief section 7)

`reports/phase16/phase16_measurement_handoff_v1.json`, **verified: true**
(every digest re-derived from bytes at write time; the baseline manifest's
library-digest binding cross-checked). It carries:

- benchmark manifest path, file sha256, manifest digest
  `ebd130198ea500248b32df990bee876583a10d53546f38a6346ec522407320c2`,
  board count 120, quick subset name;
- adversarial library path, file sha256, library digest
  `e01529cedc042e858bdf9f9036e5da48c552077dd7a5a6a1eef411393afa58e5`,
  frozen authored digest
  `dcafa1614d3a5be9867bebf7a975b984243eca9bb331f3be42cc8c571067deb3`,
  setup count 96, harvest revision 0 (empty `operator_harvest`);
- baseline pack path, sha256 and manifest digest (288 boards);
- the runner version (`phase16_measurement_runner_v1`), its entry point and
  the provider-factory interface for Agents 2/3/4;
- every baseline number above, each named with its pack; and the operator
  log schema (`phase16_operator_game_v1`) and path.

## 8. Suite discipline

Two runs, reported separately because the machine state differed:

1. **Contended run (2026-08-25, alongside a 10-worker match pack):**
   6,819 passed / **3 FAILED** / 3 skipped in 19:02. The three failures are
   all in `tests/search/test_phase12_player.py`
   (`test_tiny_search_returns_the_engines_own_decision`,
   `test_default_seed_is_deterministic_in_game_and_ply`,
   `test_search_seat_uses_the_match_seed_stream`), with a logged
   `fallback to direct … timeout at ply 24` — the accepted Phase 12 player's
   internal time cap tripping under 10-way scheduler contention, which
   converts a searched decision into a direct fallback and breaks the
   determinism assertions. These tests exercise accepted Phase 12 code that
   no Phase 16 module touches or imports.
2. **Idle re-verification (2026-08-26T05:28-05:37Z, compute lock held,
   Agent 2's next job lock-blocked at 0% CPU, no match workers):**
   - `tests/search/test_phase12_player.py` solo: **19 passed in 1.40 s** —
     all three contended failures vanish on an idle machine;
   - full suite: **6,890 passed / 3 skipped in 8:36** (green; raw outputs in
     `reports/phase16/logs/idle_phase12_player_tests.txt` and
     `idle_full_suite.txt`).

   The contention hypothesis is therefore *proved*, not assumed: the failing
   assertions go through the accepted Phase 12 player's internal time cap,
   which a 10-way-loaded scheduler can trip; no Phase 16 module imports or
   touches that code, and the tests pass idle. **The suite is green on an
   idle machine.** Lesson recorded for the phase: never run the suite
   concurrently with a match pack.

Baseline at phase start was 6,708 passed / 3 skipped; Agent 1 adds 88 tests
under `tests/evaluation/phase16/`. The counts above exceed 6,796 because
Agent 2's parallel session adds its own tests to the shared tree between
runs (6,819 at the contended run, 6,890 at the idle run); only additions
occurred — no accepted test was removed or modified.

## 9. What is and is not established

**Established:**

- The instruments exist, are frozen, digest-bound and executable: rebuilding
  every board from identity alone and comparing bytes is part of the load
  path, and tampering is refused (pinned by tests).
- The scoring runner scores Phase 15 systems and arbitrary decision-seat
  factories through one interface, resumes killed packs, and refuses the
  oracle by name.
- The operator protocol, logging schema, capture and harvest paths work end
  to end (exercised by tests and a dry-run capture; no operator game has been
  played yet).
- The baseline numbers reported above, on their named packs, with their
  game counts.

**Not established — and this report claims none of it:**

- No strength claim beyond the measured baselines. Benchmark EWRs live on
  `phase16_benchmark_v1` and adversarial EWRs on
  `phase16_adversarial_baseline_v1`; the two are never compared.
- No training conclusion. The per-family table is Agent 3's *input*, not a
  curriculum result.
- No human-strength claim: the operator re-baseline is pending; all previous
  human impressions remain retired.
- No significance claim anywhere: engineering packs, predeclared margins,
  standard errors reported.

## 10. Deviations and conservative choices

1. **T0 backup not redone** — the coordinator ran it earlier today; it was
   verified (existence, size, digest file) and recorded instead, per
   direction. Only the end-of-task refresh was executed by this agent.
2. **"120 paired boards"** — the brief's cell grid {10}x{3}x{2} yields 60
   cells; 120 boards therefore means two ordinals per cell, with ordinal 0
   doubling as the predeclared quick subset. Balance is unaffected.
3. **Seed namespace** — every new Phase 16 stream (setup draws, match seeds,
   library authoring, capture) derives via blake2b personalization
   `strat-p16m` under `phase16.agent1`, mirroring the accepted helper
   pattern. One documented exception: in-search world seeds of a Phase 16
   game come through the imported Phase 15 seat (`search_seed_for(board_id,
   ply)`), keyed on Phase 16 board ids — a property of reusing the accepted
   seat unmodified; disjoint from every Phase 15 stream because no Phase 15
   board id carries the `phase16_` prefix.
4. **Packs run uncapped** (no per-move deadline), matching Phase 15 Stage
   B/C practice; fallback counts are recorded and were zero. No latency claim
   is made from pack numbers (contended ~1.8x); the deployed caps remain the
   frozen Phase 15 candidate's idle-derived caps.
5. **Arm 3 player setup** — predeclared as the same adversarial family at
   entry offset +6 (never the opponent's own entry), so the per-family
   stratification stays meaningful in the secondary arm.
6. **Arm 1 control opponent** — drawn from the same accepted-library source
   rotation as the player's own setup, so arm 2 minus arm 1 isolates exactly
   the opponent-army distribution swap.
7. **One authoring revision before any adversarial game was played**: the
   first authored library placed the flag *last* in the fill tail, and in
   families that do not pin the flag (scout_screen, miner_wall,
   aggressive_marshal, spy_shadow) the bombs and sergeants had already filled
   the back rank — every scout_screen flag landed on the *front* rank. Legal,
   but off-signature, and it would have contaminated the measurement with an
   artifact of the generator rather than the family. The fill was corrected
   to place the flag first; the library and the baseline manifest were
   re-frozen (the digests above are the corrected ones) before the first
   baseline game; the benchmark manifest was unaffected (identical digest).
   The discarded library was never played against and appears in no result.
8. **No SeatProbe in Phase 16 packs** — the permutation-invariance and
   direct-agreement probes are Phase 15's acceptance apparatus for these
   exact seats and passed there on every arm; the Phase 16 runner reuses the
   accepted seats unchanged and keeps its packs lean. Any new seat (Agent 2's
   sampled player) should re-run its own probes before trusting pack numbers.
9. **Operator not available** — the operator-facing items ship as
   instruments: `operator_harvest` present but empty with a TODO, re-baseline
   series pending. Nothing was blocked on the operator.
10. **The full pytest suite first ran concurrently with a match pack** (the
    packs hold the compute lock for hours and the suite is not heavy
    compute). That was a mistake worth recording: the accepted Phase 12
    player tests turn out to be timing-bound through the player's internal
    time cap, and three of them failed under contention. Section 8 carries
    both the contended result and the idle re-verification; "suite green" is
    claimed only from the idle run.
11. **Lock behaviour changed mid-run from refuse to wait.** The first
    version of `acquire_lock` exited when another live pid held the lock;
    with Agent 2's chain queued on the same machine that would have silently
    skipped later packs in this agent's chain. It now waits, polling every
    60 s, which is also what overview section 5 intends. The change is to
    this agent's own untracked script and took effect between pack
    invocations; no pack was lost.
