# Phase 16 — Agent 1
## Measurement: backup, canonical benchmark, adversarial setups, operator protocol

## Mission

Build the instruments every other Phase 16 decision will be read from:

1. a safety backup of all untracked work;
2. `phase16_benchmark_v1` — the one canonical machine-opponent pack;
3. `phase16_adversarial_setups_v1` — human-realistic and operator-exploit setups,
   with a baseline measurement of how much the current system loses to them;
4. the operator series protocol and its logging, including setup harvesting.

This agent trains nothing and modifies nothing accepted. Read `00_PHASE_16_OVERVIEW.md`
first; its rules bind.

## 1. Process boundary

Same boundary as Phase 15 Agent 1 section 1: inspect Phase 14 state read-only; confirm
no learner/supervisor/collector is running (expected: stopped since 2026-08-24, only the
read-only dashboard may hold a port); never signal, edit, rotate, or finalize anything.
Record the check in `agent_01_process_boundary.json`.

## 2. Namespaces

```text
stratego/evaluation/phase16/     tests/evaluation/phase16/
scripts/run_phase16_agent01.py   scripts/phase16_capture_setup.py
data/phase16/                    reports/phase16/
```

## 3. T0 — the untracked backup (do this before anything else)

One stray `git clean` currently destroys Phases 15 and 16. Archive every untracked
path reported by `git status --porcelain` to the external volume:

```text
/Volumes/Brandon_Washington/stratego_untracked_backup_<YYYYMMDD>.tar
```

- Exclude the regenerable prefix caches: `--exclude='*_prefix_*.npy'` (~15 GB,
  rebuilt automatically by `ensure_caches`). Everything else goes in, including
  `data/phase15/` (~7.5 GB) and this instructions directory.
- Record: file list count from `tar -tf`, archive size, SHA-256, and the excluded
  paths, in `reports/phase16/agent_01_backup.json`.
- Re-run the backup (new dated file) at the end of your task so your own artifacts
  are covered. Never delete an older backup.

## 4. `phase16_benchmark_v1` — the canonical pack

Reuse Phase 15's match machinery by import (`stratego.search.phase15.matchplay`,
`positions`, `systems`); build only the pack definition and a scoring runner.

- **120 paired boards**: one per cell of {10 opponents} × {3 setup sources:
  neutral_v1, phase14_learned, targeted_family} × {2 colors}, drawn fresh from the
  accepted library's `validation` split through the orientation gate. Same opponent
  roster as Phase 15 Stage B (p18, p24, phase9_anchor, the rule-based pair, the five
  stress styles).
- Freeze a manifest with a digest: board tuples, opponents, per-decision seed
  derivation, rules version. This pack never changes; extensions get a new version.
- Deliver `score_on_benchmark(mode_or_provider, preset, workers, subset=None)` that
  can score (a) any `stratego.search.phase15.player` mode, and (b) any object
  implementing the Phase 15 decision-provider interface — so Agents 2/3/4 plug in
  without new glue. Support a predeclared 60-board quick subset for training-run
  checkpoint scoring.
- Establish baselines on the full pack: `p24_direct`, `p24_b24|TINY`,
  `p24_b24|MEDIUM`. Report per-opponent, per-source, per-color strata with counts.

## 5. `phase16_adversarial_setups_v1` — the pack that models the operator

A setup library of **8–12 named families, 96–128 setups total**, stored in the
accepted setup-library format (canonical own-orientation tuples + family metadata),
every board validated by the imported Phase 15 section-4 gate (flag row, legal rows,
exact inventory, paired-mirror check on oriented output).

Required families (author from documented human conventions and the Ataraxos paper's
setup analysis; keep each family internally varied):

```text
operator_harvest      the operator's own winning setups (see section 6 — ask)
bombed_corner_flag    flag a1/j1-corner, bombed in (the classic; ~2/3 of Ataraxos setups)
bombed_center_flag    flag mid-back-rank behind a bomb shell
scout_screen          front rank heavy with scouts, high pieces row 2-3
aggressive_marshal    marshal/general at or near the front
spy_shadow            spy adjacent to the marshal's likely path; high-bomb traps
miner_wall            miners spread wide, anti-bomb posture
decoy_flag_structure  bomb-ringed empty corner opposite the true flag
free_novelty          anything a bot would not expect; break symmetry conventions
```

**Ask the operator for 5–10 of the setups they used to beat the system**, via
`scripts/phase16_capture_setup.py` (accept a 4×10 rank grid as text, validate, append
to `operator_harvest`). If the operator is not available this session, proceed with
the authored families and leave `operator_harvest` present but empty with a TODO in
the report — do not block.

### The baseline measurement

Score with the section-4 runner, paired boards, both at TINY and MEDIUM:

```text
arm 1  benchmark control     opponent setups from the accepted library (validation)
arm 2  adversarial opponent  opponent draws from phase16_adversarial_setups_v1
arm 3  adversarial both      both sides draw from the pack (secondary)
```

vs the same opponent roster. Report overall and **per-family** EWR deltas
(arm2 − arm1) with SEs.

**Predeclared reading:** a drop ≥ 0.10 EWR confirms the distribution hypothesis and
the per-family table becomes Agent 3's training-mixture input; a drop < 0.05 weakens
it and must be stated plainly in the report (do not soften either way).

## 6. Operator series protocol and logging

Deliver `reports/phase16/operator_protocol_v1.md` plus a logging path:

- **Re-baseline series** (replaces all pre-Phase-15 human impressions, which came
  through the defective `play_phase12.py`): 10 games, operator vs
  `maximum_strength`, alternating colors, idle machine, no time pressure on the
  operator.
- **The exam** (Phase 16's acceptance test, run by Agent 5 at the end): 20 games,
  same protocol, operator free to adapt and reuse exploits. Pass:
  model EWR ≥ 0.50.
- **Logging:** every operator game appends one JSON line to
  `data/phase16/operator_games.jsonl`: timestamp, script+mode, seats, colors, both
  setups (canonical tuples + family id if drawn), full action history, result,
  ply count, per-move wall times. Build this as a thin wrapper module in
  `stratego/evaluation/phase16/operator_log.py` that `play_phase15.py`-style scripts
  can call; until Agent 2's `play_phase16.py` lands, provide
  `scripts/play_phase16_operator.py` that wraps the Phase 15 player (import, do not
  edit) with logging attached.
- **Harvest:** a utility that extracts operator setups from the log into
  `operator_harvest` (dedup by tuple).
- Report the re-baseline series result if the operator plays it during your window;
  otherwise deliver the instrument and note it pending. Track and report EWR by game
  index — the trend line separates "stronger than the operator" from "learnable by
  the operator."

## 7. Handoff

`reports/phase16/phase16_measurement_handoff_v1.json`: digests and paths for the
benchmark manifest, the adversarial library, the scoring runner version, baseline
numbers with their pack names, and the operator-log schema version. Re-verify
digests against bytes before writing `verified: true`.

## 8. Report

`reports/phase16/agent_01_report.md`, sections mirroring this file. State what is
established (instruments, baselines) and what is not (no strength claims beyond the
measured baselines; no training conclusions).
