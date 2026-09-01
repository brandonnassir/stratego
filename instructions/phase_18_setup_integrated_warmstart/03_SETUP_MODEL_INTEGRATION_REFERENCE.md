# Phase 18 - Setup Model Integration Reference

## Math, method, hypotheses, and review checklist

This reference is a compact technical companion to the Phase 18 common contract and
the source for the Phase 18 setup-integration review PDF. It is not an executable
agent instruction.

## 1. The intended mechanism

Phase 18 tests a two-part mechanism:

```text
learned setup policy
    -> selects stronger own setups
    -> supplies broad, outcome-weighted setup variation during warmstart

live setup trajectories
    -> train the Phase 8 policy/value/belief heads on unfamiliar configurations
    -> improve recognition and exploitation of unusual opponent layouts during play
```

The setup policy and C1 learner are coupled through live games, but they have different
targets:

- the setup policy receives complete-game outcomes for the setup it sampled;
- C1 receives the original Phase 8 supervised policy, W/D/L value, and hidden-piece
  belief targets reconstructed from the same live trajectories.

## 2. Notation

| Symbol | Meaning |
|---|---|
| `sigma_bar` | Complete 40-piece own setup |
| `sigma_k` | Prefix containing placements `0..k-1` |
| `t_k` | Piece type placed at prefix `k` |
| `theta_b` | Raw setup-policy snapshot that sampled the setup |
| `theta` | Current setup-policy parameters during the update |
| `pi_b` | Behavior distribution under `theta_b` |
| `pi_theta` | Current next-piece distribution |
| `z` | Setup-owner result: win `+1`, draw `0`, loss `-1` |
| `I_k` | Realized suffix negative log likelihood from prefix `k` |
| `h_k` | Predicted normalized suffix entropy, trained toward `I_k/10` |
| `alpha(n)` | Entropy regularization temperature at global iteration `n` |

## 3. Architecture

### Paper

```text
decoder-only, pre-layernorm Transformer
depth 4
embedding width 512
8 heads
feed-forward width 2,048
12.6M parameters
```

### Phase 18 scaled model

```text
decoder-only, pre-layernorm Transformer
depth 4
embedding width 128
4 heads
feed-forward width 512
802,320 parameters
```

The proportional target is approximately 740,500 parameters:

```text
paper setup parameters * project move parameters / paper move parameters
= 12.6M * 0.863959M / 14.7M
= approximately 0.7405M
```

The proposed setup model is approximately 8.3% above target and preserves the paper's
depth and four-times-width feed-forward structure.

At each prefix, the model outputs:

```text
12 piece-type logits, inventory masked
3 W/D/L logits
1 normalized conditional-entropy prediction
```

## 4. Autoregressive setup generation

Generation proceeds in canonical own-side row-major order:

```text
start -> square 0 -> square 1 -> ... -> square 39
```

At prefix `k`:

1. Count piece types already placed.
2. Mask any type whose inventory is exhausted.
3. During the handed canonical generation, mask the Flag outside the permitted
   horizontal half.
4. Sample `t_k` from the legal softmax with a recorded seed.
5. Record the full legal distribution, selected log probability, W/D/L prediction,
   and normalized entropy prediction.
6. After square 39, apply an independent seeded horizontal reflection with probability
   `0.5`.
7. Convert to Red/Blue engine orientation through the accepted boundary helper.

The reflection removes an unnecessary learning symmetry without removing either
orientation from gameplay.

## 5. Reusable pools and outcome aggregation

The raw setup model generates a pool under one immutable behavior snapshot. New games
sample from that pool. Consequently, one setup may be played in several independent
games and may remain associated with games spanning collection boundaries.

For completed outcomes `z_1 ... z_m` attributed to setup `s`:

```text
z_bar(s) = (1/m) * sum_i z_i
```

For the categorical value target:

```text
y_bar_WDL(s) = mean_i one_hot(W/D/L_i)
```

This reduces Monte Carlo variance and matches the published buffer semantics more
closely than training every unique setup from a single result.

Required attribution keys include:

```text
setup fingerprint and reflection-class fingerprint
behavior setup-model digest
pool ID and pool generation iteration
canonical and engine orientation
color and setup-owner perspective
game IDs
completed outcome count, mean, and variance
```

## 6. Conditional entropy target

The realized suffix information from prefix `k` is:

```text
I_k = -sum_{j=k}^{39} log pi_b(t_j | sigma_j)
```

It is a one-sample Monte Carlo estimate of the behavior policy's conditional suffix
entropy.

The entropy head predicts a normalized quantity:

```text
h_theta(sigma_k) approximately I_k / 10
```

with loss:

```text
L_h = mean_k (h_theta(sigma_k) - I_k/10)^2
```

Before the prediction enters the advantage it must be restored to nats:

```text
H_hat_k = 10 * h_behavior(sigma_k)
```

The centered entropy innovation is therefore:

```text
I_k - H_hat_k = I_k - 10h_behavior(sigma_k)
```

This is the central Phase 18 correction. Using `I_k - h_k` mixes raw nats with a
quantity normalized by ten and leaves a large predictable positive bonus after the
entropy head converges.

## 7. Value baseline and setup advantage

Let the behavior W/D/L probabilities at prefix `k` be:

```text
(p_win,k, p_draw,k, p_loss,k)
```

Then:

```text
V_behavior,k = p_win,k - p_loss,k
```

The setup advantage is:

```text
delta_k = (z_bar - V_behavior,k)
          + alpha(n) * (I_k - 10h_behavior,k)

alpha(n) = 0.1 / n^0.3
```

The first term favors prefixes associated with better-than-predicted outcomes. The
second is a centered entropy innovation that preserves broad setup support while the
temperature is high.

Always log both terms separately:

```text
mean and absolute mean
quantiles and extrema
entropy/outcome absolute-magnitude ratio
correlation with total advantage
```

## 8. PPO and auxiliary losses

For the chosen token:

```text
r_k = pi_theta(t_k | sigma_k) / pi_b(t_k | sigma_k)

L_PPO = -mean min(
    r_k * delta_k,
    clip(r_k, 0.8, 1.2) * delta_k
)
```

The reverse behavior KL is:

```text
L_KL = mean KL(
    pi_theta(. | sigma_k)
    ||
    pi_b(. | sigma_k)
)
```

The value loss is categorical cross-entropy against the averaged W/D/L result:

```text
L_v = mean CE(WDL_theta(sigma_k), y_bar_WDL)
```

Total loss:

```text
L_setup = L_PPO + 0.5L_v + 1.0L_h + 0.1L_KL
```

Use all prefixes and all eligible completed setups. There is no setup advantage
filter.

## 9. Optimization recipe

| Field | Value |
|---|---:|
| Learning rate | `5e-5`, constant |
| PPO clip | `0.2` |
| Effective batch | `1,024` setup episodes |
| Epochs | `5` |
| Reverse behavior KL | `0.1`, fixed |
| Value coefficient | `0.5` |
| Entropy prediction coefficient | `1.0` |
| Gradient clip | `0.5` |
| EMA | `0.999` after each complete setup update |

Raw weights generate pools. EMA weights never enter the training population and are
used only for validation, candidate export, and final evaluation.

## 10. Integration with Phase 8

### Canonical anchor stream

The exact accepted Phase 8 corpus continues to supply stable policy/value/belief
supervision. This protects elementary play and makes original held-out metrics
comparable.

### Live setup stream

```text
raw setup snapshot
    -> 1,024-setup reusable pool
    -> frozen Phase 8 teacher matchups
    -> complete trajectories
       -> Phase 8 policy targets with original teacher weights
       -> W/D/L value targets
       -> hidden-piece belief targets
       -> setup-owner completed outcomes
    -> C1 supervised update + setup PPO update
```

Only the live stream supplies setup PPO data. Treating old `neutral_v1` setups as
though the learned policy sampled them would be invalid off-policy training.

The canonical/live mixture is selected by bounded pilots. It is not derived from the
paper because the paper trains the move policy by current-policy self-play, while
Phase 18 intentionally remains a Phase 8 teacher warmstart.

## 11. Hypotheses

| ID | Hypothesis | Primary isolation | Failure meaning |
|---|---|---|---|
| H1 | Correct entropy units yield a centered residual after `h` learns | Canned parity and synthetic task | Math/data-flow defect if false |
| H2 | Handed generation plus reflection learns faster without reducing play diversity | Matched synthetic/short setup assays | Symmetry reduction not beneficial locally |
| H3 | Reusable setup pools and averaged outcomes improve signal-to-noise | Setup-only controlled comparison | Outcome variance is not the main blocker |
| H4 | The scaled 802,320-parameter setup model can rank favorable setups | Setup-only Stratego assay | Capacity, opponent distribution, or objective remains inadequate |
| H5 | Learned setups outperform the fixed setup library | C0-L minus C0-F and T-L minus T-F | Favorable setup selection not established |
| H6 | Live setup trajectories improve C1 against unseen opponent setups | T-F minus C0-F on unusual packs | Curriculum does not transfer or damages warmstart |
| H7 | The combined paired system improves without hidden regressions | T-L minus C0-F plus worst strata | Phase 18 final claim fails |

H1-H3 must be resolved before H4. H4-H5 must pass before a tandem full run. H6 and H7
must be measured separately.

## 12. Factorial evaluation

| Lane | Move model | Own setup | Opponent setup strata |
|---|---|---|---|
| C0-F | reproduced Phase 8 | fixed library | familiar + unusual |
| C0-L | reproduced Phase 8 | learned | familiar + unusual |
| T-F | setup-integrated C1 | fixed library | familiar + unusual |
| T-L | setup-integrated C1 | learned | familiar + unusual |

Key contrasts:

```text
own-setup effect under control       C0-L - C0-F
own-setup effect under tandem        T-L  - T-F
move/generalization effect           T-F  - C0-F
combined effect                      T-L  - C0-F
interaction                          (T-L - T-F) - (C0-L - C0-F)
```

Pair opponent, opponent setup, color, and seed across contrasts. Report overall and
worst-family results. The combined effect cannot compensate for a failed own-setup or
move/generalization contrast.

## 13. Main risks and required telemetry

| Risk | Required observation |
|---|---|
| Entropy term dominates outcome | Component magnitudes and ratio by iteration |
| Entropy collapses | Prefix entropy, reflection-class support, Flag/Bomb support |
| Value baseline is uncalibrated | W/D/L CE, Brier, reliability by prefix |
| Too few outcomes per setup | Count and variance distribution per setup |
| Behavior policy becomes stale | Setup policy age, PPO ratios, clip fraction, reverse KL |
| Live stream erases Phase 8 learning | Original sealed head metrics and C0 comparison |
| Curriculum specializes | Familiar versus unusual and worst-family deltas |
| Own setup masks weaker move play | Full four-lane factorial evaluation |
| Evaluator failure biases W/D/L | Explicit failed/missing/retry accounting |

## 14. Quick review checklist

Before approving a setup experiment, confirm:

```text
[ ] fresh or explicitly identified model start
[ ] behavior raw snapshot stored with every setup
[ ] inventory mask and prefix alignment tested
[ ] forced handedness plus seeded post-sample reflection
[ ] I computed as suffix NLL in nats
[ ] h target is I/10
[ ] advantage uses I - 10h
[ ] repeated outcomes aggregated per exact setup
[ ] effective batch 1,024 and five epochs
[ ] reverse KL direction is current || behavior
[ ] raw generates; EMA evaluates
[ ] canonical and live Phase 8 streams distinguished
[ ] development and sealed unusual packs do not overlap
[ ] factorial lanes isolate own-setup and move effects
[ ] practical margins and sample sizes frozen before results
[ ] next stage requires an accepted decision packet
```

## 15. References

- Sokota, Vinitsky, Hu, Kolter, and Farina, _Superhuman AI for Stratego Using
  Self-Play Reinforcement Learning and Test-Time Search_, arXiv:2511.07312v1,
  especially Sections D.2, D.3, D.6, and H.
- AtaraxosAI/stratego, published implementation, commit
  `92db29e8ffc323b1b8a2804b5c3f84695d036b05`.
- `reports/phase_8_implementation_report.md` and `reports/phase_8_data/`.
- `reports/phase17/agent_05_report.md` and
  `reports/phase17/phase17_run_closeout_v1.json`.
- `stratego/training/phase17/setup_learning.py` and related Phase 17 setup modules.
