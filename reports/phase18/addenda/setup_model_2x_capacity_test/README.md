# Operator addendum v5 — a 2× setup model: is the plateau capacity or data?

**Informal, operator-requested. Not Gate G3 evidence.** Run 2026-09-05 from the
`phase18/g3-stage6b-harness` tree; runtime under
`output/phase18/runtime/addendum_library_setup_from_scratch_2x/` (ignored; checkpoints
and 143k evaluation rows not committed).

## Question

The 1× setup model (addendum v2) reached +0.187 over its init at 1,024 periods and was
decelerating. Is that ceiling the model's capacity, or the training data — eight static
handcrafted opponents whose fighting styles it has learned to accommodate? Test: train a
model twice the size on **exactly the same games** and see whether it goes higher.

## Design

`Phase18SetupModel(blocks=8)` — 8 decoder blocks instead of 4, everything else identical:
**1,595,408 parameters vs 802,320 (1.99×)**. The trainer, recipe, learning rate, batch,
epochs, EMA, alpha schedule, library draws and match seeds are byte-identical to the 1×
run (same namespace), and the outcome tallies match the 1× run period for period —
the two models watched **the same 2,097,152 games in the same order**. No neural policy,
no co-adaptation; the eight handcrafted bots are static throughout. 1,024 periods, 9.1 h.

Evaluation as in v2: each of the 8 bots plays the frozen 2,560-game G3 schedule with its
own formation from each checkpoint's EMA (G3 seeds) or a same-family library formation;
G3 bootstrap unchanged. The library arm's rows are model-independent and were reused
from the 1× evaluation.

A memory watchdog ran throughout (orphan reaping every 60 s, checkpoint-restart on
pressure); no restart was needed. Root cause of the earlier out-of-memory crashes was
identified during this run: orphaned 12-worker C1 loader pools (~2 GB each) surviving
killed or crashed G3-style lineages.

## Result

### Absolute EWR by checkpoint, pooled over 8 handcrafted movers

| period | 1× (802k) | 2× (1.6M) | 2× − 1× | 1× gain | 2× gain |
|---|---|---|---|---|---|
| 0 | 0.3487 | 0.4107 | +0.0621 | +0.0000 | +0.0000 |
| 256 | 0.3955 | 0.4504 | +0.0549 | +0.0469 | +0.0397 |
| 384 | 0.4383 | 0.4713 | +0.0330 | +0.0896 | +0.0606 |
| 512 | 0.4717 | 0.4874 | +0.0156 | +0.1231 | +0.0766 |
| 640 | 0.4959 | 0.5029 | +0.0071 | +0.1472 | +0.0922 |
| 768 | 0.5112 | 0.5217 | +0.0105 | +0.1626 | +0.1110 |
| 896 | 0.5312 | 0.5352 | +0.0039 | +0.1826 | +0.1244 |
| 1024 | 0.5359 | 0.5429 | +0.0070 | +0.1872 | +0.1322 |
| library | 0.4989 | | | | |

**The two curves converge on the same ceiling.** The 2× model started 6.2 points higher
(its init was luckier) and finished 0.7 points higher — inside the ±0.01 noise of a single
checkpoint. It *learned less* (+0.132 vs +0.187 over its own init) because it started
closer to the same asymptote, and it too is decelerating at the end (+0.0135, +0.0077 in
its last two blocks).

| at 1,024 | 1× | 2× |
|---|---|---|
| vs handcrafted library | +0.0370 [+0.0169, +0.0576] | +0.0440 [+0.0246, +0.0644] |
| vs own init | +0.1872 [+0.1763, +0.1981] | +0.1322 [+0.1233, +0.1415] |

### Per student at 1,024

| student | 1× | 2× | 2× − 1× | library |
|---|---|---|---|---|
| basic_heuristic | 0.6479 | 0.6395 | -0.0084 | 0.5820 |
| strategic_rule_based | 0.7742 | 0.7947 | +0.0205 | 0.7498 |
| stress_berserker | 0.4668 | 0.4451 | -0.0217 | 0.4057 |
| stress_chaos | 0.3768 | 0.3713 | -0.0055 | 0.3434 |
| stress_information_miser | 0.3303 | 0.3463 | +0.0160 | 0.3223 |
| stress_miner_rush | 0.4992 | 0.5117 | +0.0125 | 0.4477 |
| stress_scout_rush | 0.4314 | 0.4680 | +0.0365 | 0.4139 |
| tactical_rule_based | 0.7605 | 0.7668 | +0.0063 | 0.7268 |

Four bots slightly favour the 2×, four the 1× — no systematic advantage.

### Entropy

The 2× model's policy entropy tracked the 1× until ~period 768, then collapsed from
1.49 to 0.42 nats/prefix (the 1× ended at 1.36). It did **not** lose
diversity: all 2,560 sampled evaluation formations at 1,024 are distinct boards and distinct
reflection classes, none dead. It became decisive about most placements, not repetitive.

## Reading

**The ceiling is in the data, not the model.** Doubling capacity on identical games
reproduced the same asymptote (~0.54 pooled, ~+4 points over the handcrafted library)
from a different starting point. Both models have learned what there is to learn about
formations *against these eight static opponents*; a larger model cannot extract more from
the same eight fighting styles. To move the ceiling, change what the model watches — more
or more varied opponents (the neural policy itself, self-play, a wider handcrafted roster)
— rather than the model.

## Caveats

One seed per model size; informal; no pre-registered rule. "2×" is depth-doubled at the
same width and learning rate; a width-doubled or re-tuned variant could learn faster, but
the question here was the ceiling, and two very different models reached the same one.
