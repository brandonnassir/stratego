# Phase 16 — Operator Series Protocol v1

_Phase 16 Agent 1, 2026-08-25. Governs every operator game from now through the
Phase 16 exam. Supersedes all pre-Phase-15 human impressions, which came through
the defective `play_phase12.py` glue (reversed Blue armies on 47/64 boards) and
are not evidence._

## 1. The two series

### Re-baseline series (any time from now; run by the operator with Agent 1's tooling)

- **10 games**, operator vs the `maximum_strength` mode (P24+B24, MEDIUM, 5.0 s cap).
- **Alternating colors**: operator plays Red in odd games (1, 3, 5, 7, 9), Blue in even.
- Series id: `rebaseline_v1`, game indices 1–10.
- Purpose: replace the old impressions with a measured machine EWR under the
  corrected orientation path. No pass mark — this is a baseline, not a gate.

### The exam (Phase 16's acceptance test; run by Agent 5 after Agents 1–4 land)

- **20 games**, same conditions, series id `exam_v1`, game indices 1–20.
- The operator may use **any legal setups** and may **adapt freely** across
  games — reusing exploits is the point of the rematch condition.
- **Pass: model EWR >= 0.50** (draws count half) over the 20 games.
- The player under exam is whatever Agent 5 designates as maximum strength at
  that point (Agent 2's stochastic player if promoted); until then this
  protocol names the frozen Phase 15 player.

## 2. Conditions (both series)

- **Idle machine**: no heavy compute during operator play. Check
  `checkpoints/phase16/COMPUTE_LOCK.json` is absent and `ps` shows no pack
  workers before starting. The read-only dashboard on port 8714 may stay up.
- **No time pressure on the operator**: the operator thinks as long as they
  like; only the machine is capped.
- One game per invocation; play games in index order; do not discard games —
  every started game is logged and counted (a resignation is a loss for the
  resigner).
- The operator may enter their own setup (`--red-setup-file` /
  `--blue-setup-file`, capture-tool grid format) or draw from the accepted
  library sources.

## 3. How to run a game

```bash
# game 1 of the re-baseline (operator = red):
.venv/bin/python scripts/play_phase16_operator.py \
    --series rebaseline_v1 --game-index 1 \
    --red human --blue maximum_strength

# game 2 (operator = blue):
.venv/bin/python scripts/play_phase16_operator.py \
    --series rebaseline_v1 --game-index 2 \
    --red maximum_strength --blue human

# bringing your own setup:
.venv/bin/python scripts/play_phase16_operator.py \
    --series rebaseline_v1 --game-index 3 \
    --red human --blue maximum_strength --red-setup-file my_setup.txt
```

Once Agent 2's `play_phase16.py` lands, it replaces this script for machine
seats; the logging module and schema stay the same.

## 4. Logging (automatic)

Every finished game appends one JSON line (schema `phase16_operator_game_v1`)
to `data/phase16/operator_games.jsonl`: timestamp, script, seats, colors, both
setups (canonical tuples + family/base id when drawn), full action history,
result, ply count, and per-move wall times for both seats. Do not edit the log;
it is append-only evidence.

## 5. Reading the series

`stratego.evaluation.phase16.operator_log.operator_series_summary(games, series)`
reports the machine's EWR overall **and by game index with a running mean**. The
trend line is the diagnostic the phase cares about:

- flat or rising machine EWR across games → stronger than the operator under
  rematch conditions;
- falling machine EWR across games → the operator is *learning* the player —
  the predictability mechanism, precisely what Agent 2's stochastic search is
  meant to remove.

Report both numbers with the series name and game count; never merge the two
series or compare either to a machine-opponent pack.

## 6. Harvesting operator setups

After any operator session:

```bash
.venv/bin/python scripts/run_phase16_agent01.py --role harvest
```

extracts every operator setup from the log into the `operator_harvest` family
of `phase16_adversarial_setups_v1` (deduplicated by canonical tuple, validated
by the same orientation gate as every authored entry; bumps
`harvest_revision`, never touches the authored families or their digest).
Setups can also be entered directly with
`scripts/phase16_capture_setup.py` (see its `--help` for the 4x10 grid format).

## 7. Status

- Instrument delivered 2026-08-25. **The operator was not available during
  Agent 1's window; the re-baseline series is pending** and `operator_harvest`
  is present but empty (TODO recorded in `agent_01_report.md`).
