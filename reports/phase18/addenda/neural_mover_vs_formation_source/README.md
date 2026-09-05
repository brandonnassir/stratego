# Operator addendum v3 — do the learned formations help the NEURAL mover?

**Informal, operator-requested. Not Gate G3 evidence.** Run 2026-09-04 from the
`phase18/g3-stage6b-harness` tree; runtime under
`output/phase18/runtime/addendum_neural_mover/` (ignored; game chunks not committed).

## Question

Addendum v2 showed a setup model trained only on handcrafted games learns formations that
beat the handcrafted library *for handcrafted movers*. Does that transfer to the neural
policy — the player Phase 18 actually cares about?

## Design

**One fixed neural mover**: the G3 control lineage's final C1 (bundle `e1c410021f71`,
C1 digest `ef8c7d29d031…`) — the policy that never co-adapted to any
learned formation, and exactly the mover that played `control_final` in G3. It plays the
frozen G3 schedule (160 held-out evaluation formations × 8 handcrafted opponents × 2
colours = 2,560 paired games) four times, differing only in where its **own** formation
comes from:

| arm | own formation |
|---|---|
| `g3_init` | the G3 init setup model `082ff778…` — this *is* `control_final`; its receipts are reused (same C1, schedule, seeds; byte-identical formations) |
| `ckpt_0` | the from-scratch model's untrained init `c549bc02…` |
| `ckpt_1024` | the from-scratch model after 1,024 periods of learning from handcrafted games only (`c0f532e7…`) |
| `library` | a same-family handcrafted library formation (rotated; never the opponent's) |

Greedy decisions, float32, battleless 200, the G3 case seeds for own-setup sampling, the
G3 paired family-stratified bootstrap — the G3 evaluation, with only the formation source
swapped. 7,680 new games, zero policy errors, every arm reconciles.

## Result

### The same mover, four formation sources

| own formation | EWR |
|---|---|
| ckpt_0 (untrained from-scratch init) | 0.4824 |
| g3_init (what the mover played in G3) | 0.6025 |
| handcrafted library | 0.6648 |
| **ckpt_1024 (learned from handcrafted games)** | **0.7324** |

### Paired contrasts (95%)

| contrast | difference |
|---|---|
| **ckpt_1024 − g3_init** | **+0.1299 [+0.1093, +0.1501]** |
| **ckpt_1024 − library** | **+0.0676 [+0.0350, +0.1015]** |
| ckpt_1024 − ckpt_0 | +0.2500 [+0.2286, +0.2714] |
| library − g3_init | +0.0623 [+0.0263, +0.0963] |
| ckpt_0 − g3_init | -0.1201 [-0.1419, -0.0983] |

Every contrast is far from zero. Handing the unchanged neural mover the learned
formations is worth **+13 points** over the formations it played in G3, and **+6.8
points** over handcrafted library formations.

### By opponent

| opponent | ckpt_1024 − g3_init | ckpt_1024 − library |
|---|---|---|
| basic_heuristic | +0.2062 | +0.0594 |
| strategic_rule_based | +0.1953 | +0.1000 |
| stress_berserker | +0.1688 | +0.0938 |
| stress_miner_rush | +0.1516 | +0.0500 |
| tactical_rule_based | +0.1156 | +0.1125 |
| stress_scout_rush | +0.1047 | +0.0703 |
| stress_chaos | +0.0750 | +0.0500 |
| stress_information_miser | +0.0219 | +0.0047 |

Gains against every opponent on both contrasts; smallest against
`stress_information_miser`, as with the handcrafted movers. Symmetric by colour
(+0.1449 red, +0.1148 blue).

## Reading

1. **Yes — the learned formations transfer to the neural mover, and the effect is large.**
   The mover was never trained on these formations (they are out of its training
   distribution) and still gains 13 points. A policy co-adapted to them might gain more.
2. **Put next to G3**: the G3 candidate's setup learning plus policy co-adaptation moved
   play by +0.3 points and FAILED the +5 margin. Formations learned purely from
   handcrafted games, with the mover untouched, move the same evaluation by +13 —
   2.6× the margin, with the interval's lower bound (+0.109) itself above it.
   This is not the G3 decision (different design, one seed, not pre-registered), but it
   locates the G3 failure precisely: formations matter a great deal to the neural mover;
   G3's setup model simply never learned good ones, because its training signal (its own
   samples) was weak and its horizon (256 periods) was about a quarter of what the
   handcrafted-game signal needed.
3. **The init lottery is real and large.** The from-scratch init is 12 points worse
   than G3's init for the same mover. Any future setup-learning design should not start
   from a random init when a library — or now a trained model — exists.

## Caveats

One seed; informal; no pre-registered rule. The mover is the control C1, not a policy
trained alongside these formations. The library arm's formations come from the same
evaluation library as the opponents' (a different formation of the same family).
