# Operator addendum v2 (extended to 1,024 periods) — a setup model trained FROM SCRATCH on handcrafted games only

**Informal, operator-requested. Not Gate G3 evidence.** Run 2026-09-04 from the
`phase18/g3-stage6b-harness` tree. The first 256 periods are reported in `README_256.md`;
this run resumed from `ckpt_0256` (full trainer state: raw weights, AdamW moments, EMA,
counters; per-period draws and match seeds keyed on the period number) and continued to
1,024. Runtime under `output/phase18/runtime/addendum_library_setup_from_scratch/` (ignored).

## Question

Does a setup model that learns *only* by watching the handcrafted bots play each other —
formations drawn from the handcrafted library — make those bots better? No neural policy
anywhere; the movers are the fixed rule-based bots throughout.

## Design (unchanged from the 256-period run)

Fresh init `c549bc02…`; the G3 setup architecture and trainer byte-for-byte. Each period
2,048 games over the 56 ordered pairs of distinct bots (no self-play), each side's formation
drawn uniformly **with replacement** from the 6,400 training-split library formations;
formations enter the buffer as teacher-forced rows scored under the current raw model;
outcomes attribute to the formation used. **1,024 periods, 2,097,152 games, 7.1 h.**

Evaluation: each of the 8 bots plays the frozen G3 schedule (160 held-out evaluation
formations × 8 opponents × 2 colours = 2,560 paired games) with its own formation from a
checkpoint's EMA model, from the init model, or a same-family library formation. G3
bootstrap, unchanged. 48 arms, zero policy errors.

## Result

### Learning curve — pooled over 8 students (20,480 paired games per contrast)

| period | EWR | vs init, 95% | vs library, 95% | entropy (nats/prefix) |
|---|---|---|---|---|
| 0 (init) | 0.3487 | — | -0.1503 [-0.1704, -0.1293] | 1.824 |
| 256 | 0.3955 | +0.0469 [+0.0379, +0.0560] | -0.1034 [-0.1238, -0.0823] | 1.701 |
| 384 | 0.4383 | +0.0896 [+0.0796, +0.1000] | -0.0606 [-0.0801, -0.0405] | 1.641 |
| 512 | 0.4717 | +0.1231 [+0.1126, +0.1337] | -0.0272 [-0.0466, -0.0071] | 1.575 |
| 640 | 0.4959 | +0.1472 [+0.1366, +0.1580] | -0.0031 [-0.0233, +0.0177] | 1.543 |
| 768 | 0.5112 | +0.1626 [+0.1528, +0.1727] | +0.0123 [-0.0069, +0.0320] | 1.494 |
| 896 | 0.5312 | +0.1826 [+0.1726, +0.1930] | +0.0323 [+0.0120, +0.0531] | 1.496 |
| 1024 | 0.5359 | +0.1872 [+0.1763, +0.1981] | +0.0370 [+0.0169, +0.0576] | 1.377 |
| library | 0.4989 | | | |

The model **crosses the handcrafted library between periods 640 and 768** and finishes
**+0.0370 [+0.0169, +0.0576] above it**, having gained **+0.1872 [+0.1763, +0.1981]**
over its init. The gain is decelerating (+4.7, +4.3, +3.4, +2.4, +1.5, +2.0, +0.5 per
128 periods), so 1,024 is near a plateau, though the entropy was still falling.

### Per student (EWR over 2,560 games)

| student | init | 256 | 1,024 | library | 1,024 − init | 1,024 − library |
|---|---|---|---|---|---|---|
| basic_heuristic | 0.4344 | 0.4820 | 0.6479 | 0.5820 | +0.2135 [+0.1902, +0.2372] | +0.0658 [+0.0298, +0.1012] |
| strategic_rule_based | 0.5021 | 0.5742 | 0.7742 | 0.7498 | +0.2721 [+0.2496, +0.2951] | +0.0244 [-0.0042, +0.0545] |
| stress_berserker | 0.2898 | 0.3342 | 0.4668 | 0.4057 | +0.1770 [+0.1535, +0.1998] | +0.0611 [+0.0224, +0.1005] |
| stress_chaos | 0.2203 | 0.2574 | 0.3768 | 0.3434 | +0.1564 [+0.1385, +0.1742] | +0.0334 [+0.0110, +0.0560] |
| stress_information_miser | 0.2283 | 0.2584 | 0.3303 | 0.3223 | +0.1020 [+0.0908, +0.1135] | +0.0080 [-0.0062, +0.0226] |
| stress_miner_rush | 0.3154 | 0.3588 | 0.4992 | 0.4477 | +0.1838 [+0.1655, +0.2025] | +0.0516 [+0.0262, +0.0771] |
| stress_scout_rush | 0.2928 | 0.3271 | 0.4314 | 0.4139 | +0.1387 [+0.1218, +0.1560] | +0.0176 [-0.0075, +0.0433] |
| tactical_rule_based | 0.5061 | 0.5721 | 0.7605 | 0.7268 | +0.2545 [+0.2335, +0.2753] | +0.0338 [+0.0035, +0.0647] |

**All eight** bots improve by 10–27 points over the init, every interval far from zero.
**All eight** now score above their handcrafted library formation; 5 of 8 individually
significant, the other 3 positive but within noise.

### Gain by opponent (1,024 − init, pooled)

| opponent | difference |
|---|---|
| stress_berserker | +0.3520 |
| basic_heuristic | +0.2365 |
| stress_chaos | +0.2330 |
| tactical_rule_based | +0.1877 |
| strategic_rule_based | +0.1809 |
| stress_scout_rush | +0.1525 |
| stress_miner_rush | +0.1510 |
| stress_information_miser | +0.0043 |

The learned formations gain against every opponent except `stress_information_miser`
(flat), and most of all against the aggressive `stress_berserker` (+35 points).

## Reading

1. **A setup model can learn, from handcrafted games alone, formations that beat the
   handcrafted library** — for every one of the eight bots, using held-out evaluation
   formations as the opponents' boards. This is the first Phase 18 result in which a
   model-generated formation beats a human-designed one.
2. **It needed a long horizon.** At G3's budget of 256 periods the model was still
   10 points below the library; it reached parity around 700 and a significant lead by
   896. The G3 pilot's 256-period horizon was far too short for this learning rule, and
   G3's own trainer, learning from the model's self-samples, never left maximum entropy
   in the same 256 periods.
3. **The signal is what the model watches.** Same loss, same recipe as G3. Learning from
   handcrafted bots playing library formations (+18.7 over init at 1,024; +4.7 at 256)
   dwarfs learning from the model's own samples (+0.95 at 256).
4. The G3 FAIL therefore looks like two compounding causes: a training signal too weak
   for the horizon, and an init lottery that starts the model deep below the library.

## Caveats

One seed; informal; no pre-registered rule. The library arm's formations come from the
same evaluation library as the opponents' (a different formation of the same family);
model arms sample one formation per case, the library arm uses a fixed one. The learned
formations are tuned to these eight bots' play; how they fare against the neural policy
or the operator is untested.
