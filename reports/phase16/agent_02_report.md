# Phase 16 — Agent 2
## Stochastic search: sample the move, sample the rollouts, measure the cost

**Selected: `stoch_t015_r100`** — tau = 0.15, tau_r = 1.0, top-p = 0.9 over the frozen **P24 + B24** system; `varied_strength` at MEDIUM, `varied_fast` at TINY.

This is an engineering deliverable, not a scientific claim. `scientific_validation_status: not performed`. No significance claim is made anywhere in this report; every table carries its own position or game count.

## 1. Process boundary and namespaces

Checked 2026-08-25T20:53:21Z: phase14_learner_or_evaluator_running = False; verdict `clear_to_run`. Method: read-only `ps` inspection; no signal, no control file, no run-state write. All Phase 16 work is additive untracked files under the Agent 2 namespaces (`stratego/search/phase16/`, `tests/search/phase16/`, `scripts/run_phase16_agent02.py`, `scripts/play_phase16.py`, `checkpoints/phase16/`, `reports/phase16/`). Heavy compute was coordinated through `checkpoints/phase16/COMPUTE_LOCK.json`.

## 2. What was built

Two independent, seed-deterministic knobs over the frozen engine (`phase12_root_world_search_v1` via the Phase 15 systems), everything else byte-identical — candidate rule, world sampling, dedup, caps, fallback, oracle refusals all unchanged:

1. **Move sampling** — `a ~ softmax(S(a)/tau)` over the existing candidate set; `tau = 0` returns the frozen argmax decision object untouched.
2. **Rollout sampling** — rollout actions for both sides drawn from the move model's legal distribution at temperature `tau_r`, restricted to the smallest set covering top-p = 0.9 mass; `tau_r = 0` **delegates to the accepted engine's own method** rather than re-implementing it.

The zero-temperature regression test (`tests/search/phase16/test_bitidentity.py`) replays frozen Phase 15 Stage A decisions from `reports/phase15/agent_02_decisions.csv` through the Phase 16 path and requires identical actions, worlds, forward counts and score margins; it was built before any diagnostic ran. Nonzero temperatures are reproducible from their seeds (`strat-p16s` streams; world seeds unchanged from the accepted Phase 15 derivation).

## 3. Stage 1 — position diagnostics (no games)

Fresh pack `phase16_agent02_positions_v1`: 120 orientation-gated replayed positions (Phase 15 pattern, observer p24, board ordinals 200+), manifest digest `6c840cf93882cbcf…`. 16 reseeded replays per arm per position; the world seed is the accepted Stage A seed, fixed, so the argmax control is constant across replays by construction.

### MEDIUM

| arm | tau | tau_r | repeat rate | entropy (nats) | agree w/ tau=0 | oracle agree | regret excess | vs control | positions x replays |
|---|---|---|---|---|---|---|---|---|---|
| stoch_t000_r000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.0000 | 0.9417 | 0.0146 | 0.0000 | 120 x 16 |
| stoch_t015_r000 | 0.15 | 0.00 | 0.6885 | 0.7909 | 0.6766 | 0.6646 | 0.0150 | 0.0004 | 120 x 16 |
| stoch_t030_r000 | 0.30 | 0.00 | 0.5255 | 1.2293 | 0.4922 | 0.4854 | 0.0194 | 0.0047 | 120 x 16 |
| stoch_t060_r000 | 0.60 | 0.00 | 0.3531 | 1.6445 | 0.3047 | 0.2969 | 0.0375 | 0.0229 | 120 x 16 |
| stoch_t000_r100 | 0.00 | 1.00 | 0.9781 | 0.0441 | 0.9635 | 0.9266 | 0.0175 | 0.0029 | 120 x 16 |
| stoch_t015_r100 | 0.15 | 1.00 | 0.6807 | 0.8108 | 0.6641 | 0.6609 | 0.0147 | 0.0001 | 120 x 16 |
| stoch_t030_r100 | 0.30 | 1.00 | 0.5193 | 1.2448 | 0.4807 | 0.4729 | 0.0177 | 0.0030 | 120 x 16 |
| stoch_t060_r100 | 0.60 | 1.00 | 0.3609 | 1.6348 | 0.3172 | 0.3146 | 0.0301 | 0.0155 | 120 x 16 |

Oracle regret floor at MEDIUM: **0.0514** (the oracle's own S-selection regret; excess columns are read against it, exactly as the Phase 15 mixture pilot reads them).

### TINY

| arm | tau | tau_r | repeat rate | entropy (nats) | agree w/ tau=0 | oracle agree | regret excess | vs control | positions x replays |
|---|---|---|---|---|---|---|---|---|---|
| stoch_t000_r000 | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.0000 | 0.9250 | 0.0115 | 0.0000 | 120 x 16 |
| stoch_t015_r000 | 0.15 | 0.00 | 0.6859 | 0.7937 | 0.6708 | 0.6594 | 0.0072 | -0.0043 | 120 x 16 |
| stoch_t030_r000 | 0.30 | 0.00 | 0.5234 | 1.2210 | 0.4922 | 0.4870 | 0.0144 | 0.0029 | 120 x 16 |
| stoch_t060_r000 | 0.60 | 0.00 | 0.3542 | 1.6352 | 0.3083 | 0.3047 | 0.0295 | 0.0180 | 120 x 16 |
| stoch_t000_r100 | 0.00 | 1.00 | 0.9865 | 0.0352 | 0.9865 | 0.9229 | 0.0113 | -0.0002 | 120 x 16 |
| stoch_t015_r100 | 0.15 | 1.00 | 0.6792 | 0.8160 | 0.6641 | 0.6536 | 0.0100 | -0.0015 | 120 x 16 |
| stoch_t030_r100 | 0.30 | 1.00 | 0.5224 | 1.2352 | 0.4870 | 0.4818 | 0.0139 | 0.0024 | 120 x 16 |
| stoch_t060_r100 | 0.60 | 1.00 | 0.3630 | 1.6378 | 0.3182 | 0.3146 | 0.0246 | 0.0131 | 120 x 16 |

Oracle regret floor at TINY: **0.0484** (the oracle's own S-selection regret; excess columns are read against it, exactly as the Phase 15 mixture pilot reads them).

### The predeclared filter

> an arm survives if its mean oracle Q-regret excess is within +0.01 of the tau=0 control at every Stage 1 budget; the control survives by definition; the full grid is reported regardless

Survivors (margin +0.01): `stoch_t000_r000`, `stoch_t000_r100`, `stoch_t015_r000`, `stoch_t015_r100`, `stoch_t030_r000`, `stoch_t030_r100`.
Eliminated: `stoch_t060_r000`, `stoch_t060_r100`.

## 4. Stage 2 — the match pack

Boards: `phase16_agent02_interim_pack_v1` (source: interim_fallback, 60 paired boards). 720 games recorded. Same accepted seed streams per board and ply for every arm; paired against the `stoch_t000_r000` control.

### MEDIUM

| arm | W/D/L | EWR | paired vs control | worst opponent | median s/move (pack) | fallbacks | games |
|---|---|---|---|---|---|---|---|
| stoch_t000_r000 @ MEDIUM | 48/1/11 | 0.8083 | - | 0.5833 (p24, 6 games) | 3.087 | 0 | 60 |
| stoch_t000_r100 @ MEDIUM | 48/3/9 | 0.8250 | +0.0167 ± 0.0460 | 0.3333 (p24, 6 games) | 3.101 | 0 | 60 |
| stoch_t015_r000 @ MEDIUM | 45/3/12 | 0.7750 | -0.0333 ± 0.0593 | 0.2500 (p24, 6 games) | 3.082 | 0 | 60 |
| stoch_t015_r100 @ MEDIUM | 49/1/10 | 0.8250 | +0.0167 ± 0.0640 | 0.3333 (p24, 6 games) | 3.072 | 0 | 60 |
| stoch_t030_r000 @ MEDIUM | 43/1/16 | 0.7250 | -0.0833 ± 0.0631 | 0.4167 (strategic_rule_based, 6 games) | 3.060 | 0 | 60 |
| stoch_t030_r100 @ MEDIUM | 43/0/17 | 0.7167 | -0.0917 ± 0.0819 | 0.5000 (p18, 6 games) | 3.067 | 0 | 60 |

60 paired boards per arm; draws count half.

### TINY

| arm | W/D/L | EWR | paired vs control | worst opponent | median s/move (pack) | fallbacks | games |
|---|---|---|---|---|---|---|---|
| stoch_t000_r000 @ TINY | 49/1/10 | 0.8250 | - | 0.2500 (p24, 6 games) | 0.448 | 0 | 60 |
| stoch_t000_r100 @ TINY | 49/2/9 | 0.8333 | +0.0083 ± 0.0483 | 0.4167 (p24, 6 games) | 0.450 | 0 | 60 |
| stoch_t015_r000 @ TINY | 43/1/16 | 0.7250 | -0.1000 ± 0.0529 | 0.0833 (p24, 6 games) | 0.446 | 0 | 60 |
| stoch_t015_r100 @ TINY | 48/1/11 | 0.8083 | -0.0167 ± 0.0594 | 0.5000 (p24, 6 games) | 0.451 | 0 | 60 |
| stoch_t030_r000 @ TINY | 49/0/11 | 0.8167 | -0.0083 ± 0.0483 | 0.1667 (p24, 6 games) | 0.444 | 0 | 60 |
| stoch_t030_r100 @ TINY | 40/0/20 | 0.6667 | -0.1583 ± 0.0576 | 0.1667 (p24, 6 games) | 0.448 | 0 | 60 |

60 paired boards per arm; draws count half.

### The predeclared selection

> among arms with EWR within 0.05 of the control at MEDIUM, select the one with the lowest Stage 1 repeat rate (ties: lower tau, then lower tau_r); if none qualifies, select tau=0.15, tau_r=0.0 if it is within 0.05; else report no-viable-stochastic-mode and keep argmax

Control EWR at MEDIUM: **0.8083**. Qualifiers (within 0.05): `stoch_t000_r100`, `stoch_t015_r000`, `stoch_t015_r100`. **Selected: `stoch_t015_r100`** — lowest Stage 1 repeat rate at MEDIUM among arms with EWR within 0.05 of the control.

**How to read the two budgets.** MEDIUM is the deciding pack: it is the maximum-strength budget the selection rule names and the one `varied_strength` ships at. TINY is supporting evidence on the same boards, and it does not contradict the choice — the selected arm sits 0.017 above the control at MEDIUM and 0.017 below it at TINY, both well inside a 60-board pack's paired standard error (0.048-0.064). What TINY adds is the same ordering signal at the extremes: the two `tau = 0.30`-plus arms and `tau_r`-only sampling behave consistently across budgets.

**Absolute EWRs belong to their pack.** Every number in this section is measured on `phase16_agent02_interim_pack_v1` and may not be compared across packs. Agent 1 measured a sign flip of exactly this kind on its own instrument (TINY search below direct on `phase16_benchmark_v1`, the reverse of Phase 15's reading), which is why the selection rule is written on **paired deltas against the tau = 0 control on identical boards** rather than on absolute strength.

### The adversarial delta (brief section 4, last bullet)

Opponent-side setups from `phase16_adversarial_baseline_v1` (digest `e937df0dc395462f…`), 96 paired boards at MEDIUM:

| arm | W/D/L | EWR | paired vs control (same boards) | games |
|---|---|---|---|---|
| stoch_t000_r000 | 77/4/15 | 0.8229 | - | 96 |
| stoch_t015_r100 | 81/1/14 | 0.8490 | +0.0260 ± 0.0394 | 96 |

This is the brief's opponent-side adversarial check: the selected arm and the deterministic control on identical boards from Agent 1's frozen pack. The point estimate favours the stochastic arm; on 96 paired boards it is inside one standard error, so it is recorded as *not a regression* rather than as a gain.

## 5. Repeat-encounter probe (recorded, not gating)

80 games at MEDIUM: the selected arm (`stoch_t015_r100`) and the control (`stoch_t000_r000`), 20 sequential games vs each of p18 and p24 direct, a fresh board per game (ordinals 300+).

| arm | games | EWR | slope per game index | first half | second half |
|---|---|---|---|---|---|
| stoch_t000_r000 | 40 | 0.6250 | 0.01880 | 0.5500 (20) | 0.7000 (20) |
| stoch_t015_r100 | 40 | 0.6875 | 0.00583 | 0.6500 (20) | 0.7250 (20) |

**Caveat, stated plainly:** fixed bots cannot adapt across games, so a flat trend here is a weak proxy only; the real adaptation test is the operator series.

## 6. `scripts/play_phase16.py` and the caps

The CLI supersedes `play_phase15.py` (which stays untouched): all Phase 15 modes by import, plus `varied_strength` (selected configuration at MEDIUM) and `varied_fast` (same configuration at TINY). Operator logging goes through Agent 1's `stratego.evaluation.phase16.operator_log` when that module is present, else a local JSONL fallback with the same schema, to `data/phase16/operator_games.jsonl`. Same information boundary as Phase 15: legal knowledge only; the oracle is refused by name and by absence from every mode table.

Idle latency, measured one process, 1 torch thread(s), idle machine; full varied-mode decisions (search + one softmax draw) on 40 replayed diagnostic positions:

| preset | median s/move | p95 | max | forwards/move | cap |
|---|---|---|---|---|---|
| MEDIUM | 1.715 | 1.760 | 1.777 | 1928.3 | 5.00s |
| TINY | 0.247 | 0.252 | 0.288 | 287.6 | 0.91s |

Cap rule: keep the frozen Phase 15 caps unless idle p95 grew by more than 10%, then min(3.5 x p95, 5.0). Caps changed: no — the Phase 15 caps stand.
Stage 1/2 move times are pack numbers under worker contention (~1.8x inflated, Phase 15 measured it) and are never used for caps.

## 7. Candidate freeze and handoff

`checkpoints/phase16/phase16_stochastic_candidate_v1.json` binds the selected configuration to the frozen bytes: P24 `622d9e6caa72…`, B24 `ac5e15b87f5c…`, applied belief temperature 1.0, the accepted search configuration, the budgets, the idle-measured caps, the Stage 1/2 headline numbers with their pack names, and the known limitations. `oracle_available_in_production = false`.

## 8. Known limitations

- machine packs cannot measure adaptation resistance: every Stage 1/2/probe opponent is a fixed policy that cannot learn the player's habits, so the unpredictability these numbers buy is only measurable in the operator exam (Agent 1's protocol, Agent 5's exam)
- a compact engineering pack: 60 paired boards per arm per budget in Stage 2; no significance claim is made anywhere
- the repeat-encounter probe uses fixed bots and is a weak proxy by construction; a flat trend there does not demonstrate adaptation resistance
- Stage 1 shares searches across move-temperature arms at the same tau_r by design (paired draws on identical score vectors); arm rows are correlated across tau, which pairing exploits and independence-based readings must not assume
- with sampled rollouts, deduplicated duplicate worlds share one sampled rollout (the accepted dedup is kept byte-identical); this changes Q variance, not support
- no scientific validation phase was performed; this is an engineering selection

### Deviations, recorded

- Stage 1 positions use one observer (P24, the selected system's move model) instead of Phase 15's two-observer rotation; conservative match to the single pairing under study
- Stage 1 move-temperature arms at the same tau_r share the sixteen underlying searches (rollout stream keyed by the rollout configuration only), so tau differences are pure move-sampling effects on identical score vectors
- the Stage 1 survival filter is applied at both budgets (the conservative reading of the brief's single-margin rule), declared before any number was seen

## 9. Suite, CLI verification, and reproduction

Full pytest suite: **7004 passed / 3 skipped / 0 failed** in 9.0 min (2026-08-26T16:19:09Z), run idle machine, single process, under the compute lock; baseline at phase start was 6,708 / 3 — only additions. The three timeout-sensitive `tests/search/test_phase12_player.py` failures seen earlier under ten-way pack contention do **not** reproduce idle: that file passes 19/19 here, so the cause was scheduler contention tripping a deadline, not a defect this agent introduced.

CLI verified end-to-end on an idle machine: `scripts/play_phase16.py --red varied_fast --blue varied_strength --setup-seed 3` played a complete game (169 plies, result red (flag_capture)), 169 varied-mode decisions, 53 of them sampled away from the argmax, no fallbacks ({}), and appended one `phase16_operator_game_v1` line to `data/phase16/operator_games.jsonl`.

Every stage re-runs from its role: `run_phase16_agent02.py --role positions | stage1 | stage2 | probe | benchscore | latency | candidate | report` (stage1/stage2 take `--budget TINY|MEDIUM`; stage2, probe and benchscore resume from their JSONL; every heavy role takes `checkpoints/phase16/COMPUTE_LOCK.json` and accepts `--wait-lock <minutes>`).

Control arm: `stoch_t000_r000` — the frozen Phase 15 selection, reached through the accepted builders, playing the accepted argmax.
