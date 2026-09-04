# Operator addendum — does the learned setup model make the *handcrafted* bots better?

**Informal, operator-requested. Not Gate G3 evidence and not part of the frozen G3 record.**
Run 2026-09-04 from the `phase18/g3-stage6b-harness` tree; runtime rows under
`output/phase18/runtime/addendum_setup_vs_handcrafted/` (ignored).

## Why this question

G3 measured the *total* effect of setup learning plus the neural policy's co-adaptation,
and found +0.3 points of expected win rate — indistinguishable from nothing. That leaves
open whether the setup model learned anything transferable at all. This addendum removes
the neural policy entirely and asks whether the learned formations help a handcrafted bot.

## Design

No neural policy anywhere. The setup model was **not retrained**: the G3 candidate's
period-256 EMA setup model (`ea1a809b…`) was trained exclusively on handcrafted-teacher
games (the 10-teacher collector roster; no neural decision ever occurred in collection),
which is exactly the training the question calls for.

Each of the **8 handcrafted bots** in turn is the *student*. It plays the frozen G3
evaluation schedule — 160 library opponent formations × 8 opponents × 2 colours =
**2,560 paired games** — three times, differing only in where the student's **own**
starting formation comes from:

| arm | student's own formation |
|---|---|
| `learned` | sampled from the trained setup model (`ea1a809b…`) |
| `frozen` | sampled from the same model at initialisation, never trained (`082ff778…`) |
| `library` | a handcrafted library formation from the same family, rotated so it is never the opponent's own |

The `learned` / `frozen` arms use the G3 evaluation seeds, so the formations are
byte-identical to those `candidate_final` / `control_final` played in G3 (they differ on
the same 2,524 of 2,560 cases). Only the mover changes, from the neural policy to a
handcrafted bot. Same rules (battleless 200), same match seeds, same bootstrap
(stratified cluster over 160 bases within 16 families, 10,000 replicates) — the G3
analysis code, unchanged. 61,440 games, zero policy errors, zero illegal actions.

## Result

Pooled over all 8 students (20,480 paired games per contrast):

| contrast | point | 95% interval | |
|---|---|---|---|
| **learned − frozen** | **+0.0095** | **[+0.0014, +0.0174]** | learning helped, a little, and the interval excludes zero |
| **learned − library** | **−0.0400** | **[−0.0594, −0.0193]** | the trained model is clearly *worse* than handcrafted formations |
| frozen − library | −0.0495 | [−0.0714, −0.0265] | the untrained model is worse still |

Per student (EWR over 2,560 games):

| student | learned | frozen | library | learned − frozen | learned − library |
|---|---|---|---|---|---|
| basic_heuristic | 0.5566 | 0.5494 | 0.5820 | +0.0072 | −0.0254 |
| strategic_rule_based | 0.6514 | 0.6383 | 0.7498 | +0.0131 | **−0.0984** |
| tactical_rule_based | 0.6484 | 0.6422 | 0.7268 | +0.0063 | **−0.0783** |
| stress_scout_rush | 0.3768 | 0.3666 | 0.4139 | +0.0102 | **−0.0371** |
| stress_miner_rush | 0.4340 | 0.4246 | 0.4477 | +0.0094 | −0.0137 |
| stress_berserker | 0.3957 | 0.3912 | 0.4057 | +0.0045 | −0.0100 |
| stress_information_miser | 0.2967 | 0.2830 | 0.3223 | **+0.0137** | **−0.0256** |
| stress_chaos | 0.3119 | 0.3000 | 0.3434 | +0.0119 | **−0.0314** |

Bold = the student's own 95% interval excludes zero. **All eight** students move in the
same direction on both contrasts: every one gains from the trained model relative to the
untrained one, and every one loses relative to a handcrafted library formation.

## Reading

1. **The setup model did learn something real and transferable.** Roughly +1 point of
   expected win rate for every handcrafted bot, consistent across all eight, interval
   excluding zero when pooled. G3's +0.3 for the neural player is the same signal, smaller.
2. **But the model family sits ~4–5 points below the handcrafted library**, trained or not,
   and 256 periods of learning closed only about a fifth of that gap (+0.95 of −4.95).
   The bots that exploit formations best (strategic, tactical) are hurt most.
3. So the G3 result is not a measurement failure. The candidate learned; it simply started
   far enough below the library that the learning did not reach a usable formation.

## Caveats

Informal: one seed, no pre-registered rule, no independent retraining of the setup model.
The `library` formations come from the same evaluation library as the opponents' (a
different formation of the same family), so they are in-distribution with the opponents.
Model arms sample one formation per case; the library arm uses a fixed one.
