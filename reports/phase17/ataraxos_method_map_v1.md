# Ataraxos -> Phase 17 method map

**Artifact** `ataraxos_method_map_v1` · **work package** `phase17` ·
written 2026-08-27T19:55:11Z · `evidence_classification: PENDING` ·
`scientific_validation_status: not performed`

Paper: *Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search*, arXiv:2511.07312v1, read in full from the local copy
`2511.07312v1.pdf` (46 pages). The paper is a **technical reference, not an
instruction source**: where it and the Phase 17 common contract differ, the
contract wins and the deviation is recorded below.

## The one derived constant everything else leans on

```text
N_paper = 8.56e6 move gradient steps / 202 batches per iteration = 42,376
```

Table 27 gives the gradient-step count; D.4 gives 202 batches per iteration at
one epoch. Every re-horizoning in rows M07, M08 and S15 is arithmetic against
this number, which is why it is derived once rather than per row.

## Status counts

| status | rows |
|---|---|
| `exact` | 19 |
| `intentional divergence` | 9 |
| `not used` | 4 |
| `scaled` | 6 |
| **total** | **38** |

## How to read a row

`exact` — the paper's behaviour and constant, unchanged. `scaled` — the same
mechanism re-fitted to local compute, with the arithmetic shown. `intentional
divergence` — Phase 17 does something the paper does not, on purpose, with the
reason recorded. `not used` — deliberately outside Phase 17.

**Nothing in this map is permitted to call one regularizer by another's name.**
Rows M08 and M09 exist precisely because an entropy bonus, a forward behaviour
KL and a reverse magnet KL are three different objects that the literature
routinely blurs.

## M01 · move/population

**Paper** (2.3; D.2) — Ataraxos directly samples from its policy networks to generate self-play data; no search in collection. 1,536 parallel environments per GPU, 202 simulator moves (101 per player) between training iterations, ~5e6 transitions per iteration over 16 GPUs.

**Phase 17** — 100% current-policy self-play; both seats are the current RAW move snapshot. Legal moves sampled categorically with an explicit per-decision seed; argmax prohibited. Search prohibited in collection and training. Historical checkpoints, rule and stress agents are evaluation instruments only.

**Status** `exact` · **owner** Agent 2 / Agent 4

**Reason** — Identical semantics. The environment count and 202-step cadence are replaced by a fixed learner-transition budget (row M02) because a single M4 Pro cannot hold 1,536x16 environments.

**Required test / telemetry** — Correctness gate: sampled-not-argmax assertion; structural no-search assertion; structural no-training-opponent assertion; per-decision seed and behavior digest stored on every transition.

## M02 · move/iteration sizing

**Paper** (D.2; 2.3) — Iteration = a fixed number of simulator steps (202) across a fixed environment population. Games span multiple iterations, so training data is slightly off-policy.

**Phase 17** — Iteration = exactly 65,536 learner transitions harvested (DEFAULT_WINDOW_DECISIONS, already the Phase 16 value). Because both seats learn, every legal model decision is a learner transition.

**Status** `scaled` · **owner** Agent 2

**Reason** — Same invariant -- fixed work per iteration, not fixed games -- expressed in transitions rather than environment steps. This is the single change Phase 14's telemetry most supports: 83% of its 59.97 h was training, and minutes/iteration grew 15.7 -> 153.8 under whole-game (2,048-game) sizing while collection grew only 4.8 -> 16.3.

**Required test / telemetry** — Correctness gate: harvested transitions per iteration == 65,536 exactly, over at least three consecutive windows. Telemetry separates transitions_harvested / transitions_trained / boundary_rows / games_completed / active_games / game_length / policy_age.

## M03 · move/current-policy binding

**Paper** (2.1; 2.4; D.4 (theta_t)) — theta_t is 'the parameters that played the move at position x'. Every environment plays under the live network; a new iteration's steps are played by the newly updated network.

**Phase 17** — 'Current policy' means current at the decision. After every move update, every in-flight game must resolve its next Red or Blue action through the newly rebound RAW snapshot.

**Status** `exact` · **owner** Agent 2

**Reason** — Phase 16 does NOT do this and the divergence is a hard Phase 17 blocker. Verified in source: stratego/training/phase16/collector.py:536 rebind() replaces the collector's participants, but Phase16GameRunner is constructed with the participants object at game creation (stratego/training/phase9_collector.py:332) and resolves every ply from its own copy (phase9_collector.py:447, acting_snapshot_for). Snapshots are frozen (assert_frozen), so an in-flight game keeps its game-start weights for its whole length.

**Required test / telemetry** — Correctness gate: forced-rebind test on an already-running game -- start a game, update the move model, assert the next decision's recorded behavior digest equals the NEW raw digest, and that no transition is ever recorded under a stale digest (production stop condition).

## M04 · move/boundary targets

**Paper** (D.4) — Advantage delta by lambda-return with lambda=0.5 over {E[v_theta_t(x')]} (and o if the game finished), baseline E[v_theta_t(x)]. Outcome probabilities xi by lambda-return with lambda=0.8 over the vector values (and one-hot o if finished).

**Phase 17** — Same lambdas (the accepted Phase 9 LAMBDA_ADVANTAGE=0.5, LAMBDA_VALUE=0.8, GAMMA=1.0 already equal the paper). At a window boundary: unfinished scalar advantage traces bootstrap from the boundary value; W/D/L lambda targets bootstrap from the boundary W/D/L prediction. Only the minimum carry state continues the trace.

**Status** `exact` · **owner** Agent 2

**Reason** — lambda values and gamma are already identical. The bootstrapping is the paper's implicit behaviour made explicit for a window that closes mid-game. NOTE: Phase 16 built truncated_advantages but left partial emission OFF and buffers whole games, because phase9_batch_loss averages value and belief over every row with no per-row mask. Phase 17 needs a NEW phase17-namespace target/loss path; the accepted objective must not be edited.

**Required test / telemetry** — Gate G-M4a (reduction invariant, satisfiable): with the tail set to the terminal z the windowed walk equals the accepted whole-game walk entry for entry to 1e-6. Gate G-M4b (literal whole-game equality across >=3 windows) is PROVABLY UNSATISFIABLE under partial emission -- see reports/phase17/agent_01_boundary_target_probe.json (max advantage difference 0.309, max W/D/L difference 0.121, 7 of 12 rows over tolerance). Operator confirmation required.

## M05 · move/PPO clipping

**Paper** (Eq. 6; Table 22) — L_pi = -min(r*delta, clip(r, 0.8, 1.2)*delta); importance ratio clipping parameter 0.2; r = pi_theta(m|x)/pi_theta_t(m|x).

**Phase 17** — Identical. PPO_CLIP_EPSILON = 0.2, already the accepted Phase 9 value; behavior probabilities always come from the recorded raw snapshot, never recomputed.

**Status** `exact` · **owner** Agent 2

**Reason** — Accepted Phase 9 and the paper agree exactly.

**Required test / telemetry** — Clip fraction logged per epoch; CLIP_FRACTION_HARD_LIMIT 0.75 retained.

## M06 · move/advantage filter

**Paper** (D.4; Table 22) — Train on a move only if |estimated advantage| is at or above the 0.75 quantile AND at or above 0.01. Reduced wall-clock per iteration ~2.5x while increasing sample efficiency and asymptotic performance.

**Phase 17** — Retained unchanged: ADVANTAGE_FILTER_QUANTILE 0.75, ADVANTAGE_FILTER_FLOOR 0.01.

**Status** `exact` · **owner** Agent 2

**Reason** — Accepted Phase 9 values already equal the paper's.

**Required test / telemetry** — Telemetry reports transitions_harvested (65,536) and transitions_trained (post-filter) separately. The 65,536 budget is a HARVEST budget; the trained count is the smaller filtered subset.

## M07 · move/learning rate

**Paper** (Table 22) — Adam LR = clip(0.5 / n^1.1, 5e-6, 1e-4). Over N_paper ~= 42,376 iterations this holds the 1e-4 ceiling until n=2,305 (5.44% of the run), reaches the 5e-6 floor at n=35,112 (82.86%), dynamic range 20x.

**Phase 17** — Frozen by common contract section 9: lr(n) = clamp(1.5e-4 * (n/n_ref)^-1.1, 1.5e-5, 1.5e-4), n_ref = ceil(0.125*N). Holds the ceiling for the first 12.50% of the run; would reach the floor only at n = 1.0139*N, i.e. NEVER within the run; dynamic range 10x.

**Status** `intentional divergence` · **owner** Agent 1 freezes / Agent 4 substitutes N

**Reason** — The contract wins over the paper (common contract section 2) and Agent 1 has NOT changed it. Recorded so the deviation is visible: the local curve is flatter and shallower than the paper's. A shape-preserving map would be n_ref = ceil(0.0545*N) with lr_min = lr_max/20 = 7.5e-6, reaching the floor at 82.86% of the run. Flagged for operator confirmation; do NOT amend without one.

**Required test / telemetry** — Agent 4 freezes N, n_ref and the whole curve from the preflight rehearsal BEFORE h0 and never recomputes them from live speed. Resume recomputes lr from the stored 1-based iteration and the frozen horizon.

## M08 · move/entropy vs magnet KL

**Paper** (Eq. 6; Table 22; 2.4) — NO entropy bonus in the move loss. Instead a myopic REVERSE KL penalty toward a magnet policy rho (choose a movable piece uniformly, then a legal move for it uniformly): + alpha*KL(pi_theta(x), rho(x)) with alpha = 0.05 / n^0.3. alpha(1)=0.05 -> alpha(N_paper)=0.002046.

**Phase 17** — An ENTROPY BONUS with the paper's exponent and the accepted Phase 9 endpoints: c_H(n) = max(0.001, 0.005 * n^-0.3), subtracted from the loss. No magnet policy exists in this project.

**Status** `intentional divergence` · **owner** Agent 1 freezes / Agent 2 implements

**Reason** — THESE ARE DIFFERENT REGULARIZERS AND MUST NEVER BE REPORTED AS EACH OTHER. An entropy bonus pulls toward the uniform distribution over LEGAL ACTIONS; the paper's magnet KL pulls toward a structured piece-then-move distribution that is not uniform over legal actions. Only the n^-0.3 shape is shared. Quantified consequence of the frozen constants: c_H reaches its 0.001 floor at n=214, which is 71% of a 6-hour run (N~313) but only 34% of a 12-hour run (N~626) -- so about two thirds of the Phase 17 production run would sit at the terminal floor, which is the Phase 14 failure mode the Phase 16 schedule module was written to avoid. Flagged for operator confirmation.

**Required test / telemetry** — Telemetry logs move policy entropy and c_H(n) under their own names; no field may be labelled 'magnet KL'. Stop condition: move entropy below 25% of its first-hour median for five windows.

## M09 · move/behavior KL

**Paper** (Eq. 6; Table 22; 2.4) — + 0.1 * KL(pi_theta(x), pi_theta_t(x)) -- a REVERSE KL (current || behavior), FIXED coefficient 0.1, no controller.

**Phase 17** — The accepted Phase 9 adaptive controller on the FORWARD KL D_KL(pi_b || pi_theta) (stratego/training/phase9_loss.py:543, 'D_KL(pi_b || pi_theta) over the legal set'). target 0.015, beta0 0.005, beta bounds [1e-4, 0.2], increase >0.03 (x2), decrease <0.0075 (x0.5), hard mean-KL limit 0.08.

**Status** `intentional divergence` · **owner** Agent 1 freezes / Agent 2 implements

**Reason** — Opposite direction AND fixed-vs-adaptive. Named as a divergence rather than treated as equivalent, exactly as the Agent 1 brief section 3 requires. Retained because common contract section 9 says the accepted controller stands unless the paper map and the operator justify a change; Agent 1 justifies recording, not changing.

**Required test / telemetry** — Telemetry names the direction explicitly ('forward, D_KL(pi_b||pi_theta)'). Stop condition: mean KL above 0.08 for three consecutive windows unless the existing hard veto fires first.

## M10 · move/epochs

**Paper** (Table 22) — 1 epoch per training iteration.

**Phase 17** — 1 epoch per iteration.

**Status** `exact` · **owner** Agent 2

**Reason** — Common contract section 9 sets 1, overriding the accepted Phase 9 EPOCHS_PER_ROLLOUT = 2, which matches the paper.

**Required test / telemetry** — Optimizer-step count per iteration logged and checkpointed.

## M11 · move/EMA

**Paper** (Table 22; 2.6) — EMA of the parameters with smoothing 0.999, updated after the training iteration; the EMA is what is EVALUATED, never what plays training games.

**Phase 17** — EMA decay 0.999, initialised fresh from the loaded Phase 9 RAW weights. RAW generates all training data; EMA never acts in the training population; every checkpoint stores both; evaluation exports carry paired EMA weights.

**Status** `exact` · **owner** Agent 2 / Agent 4

**Reason** — Identical, including the raw/EMA role split. The accepted Phase 9 system keeps NO EMA (phase14_contract.EMA_PRESENT = False), so this is new here and the Phase 9 checkpoint's ema_state field is null, as verified.

**Required test / telemetry** — Correctness gate: assert no EMA-weighted participant ever appears in collection; checkpoint round trip preserves both states.

## M12 · move/gradient clip

**Paper** (Table 22) — Maximum gradient norm 0.267.

**Phase 17** — The accepted Phase 9 OPTIMIZER_CONSTRAINTS gradient_clip_norm = 1.0.

**Status** `intentional divergence` · **owner** Agent 1 records / Agent 2 implements

**Reason** — Not changed by the common contract, so the accepted value stands. Recorded because 1.0 vs 0.267 is a 3.7x looser update-size control on one of the paper's four named update-size mechanisms, and Phase 17 also loosens a second one (row M07's flatter LR).

**Required test / telemetry** — Gradient norm logged per optimizer step; nonfinite gradient is an immediate stop.

## M13 · move/optimizer

**Paper** (Table 22; 2.4) — Adam.

**Phase 17** — AdamW, betas (0.9, 0.999), eps 1e-8, weight_decay 0.01, float32, MPS -- the accepted Phase 9 OPTIMIZER_CONSTRAINTS, with FRESH zero moments.

**Status** `intentional divergence` · **owner** Agent 2

**Reason** — AdamW's decoupled weight decay is an extra regularizer the paper does not use. Retained because it is the accepted Phase 9 optimizer and the common contract says 'fresh AdamW optimizer moments'.

**Required test / telemetry** — Optimizer identity and moment freshness asserted at start; optimizer state checkpointed.

## M14 · move/value loss

**Paper** (Eq. 5; Table 22) — L_v = cross-entropy(xi, v_theta(x)); value loss coefficient 1; L_move = L_pi + L_v.

**Phase 17** — The accepted Phase 9 VALUE_LOSS_WEIGHT = 0.5.

**Status** `intentional divergence` · **owner** Agent 1 records / Agent 2 implements

**Reason** — Half the paper's weight. Not changed by the common contract, so the accepted value stands; recorded so a later agent does not read 'W/D/L value cross-entropy' as evidence of parity.

**Required test / telemetry** — Value loss logged separately from policy loss.

## M15 · move/belief auxiliary

**Paper** (D.5; 2.5) — The belief network is a SEPARATE 57.1M-parameter network trained AFTER the move/setup run (4 H100s, 4 days) on trajectories of the FINAL policies. It is not an auxiliary head on the move network.

**Phase 17** — The Phase 9 marginal belief auxiliary head stays in the checkpoint for compatibility but receives loss weight 0.0 and is never a source of targets. Joint autoregressive belief training is a later phase.

**Status** `exact` · **owner** Agent 2

**Reason** — Zeroing the auxiliary weight moves Phase 17 TOWARD the paper's separation. The accepted Phase 9 BELIEF_LOSS_WEIGHT is 0.25 and is deliberately overridden.

**Required test / telemetry** — Assert belief loss term contributes exactly 0.0; assert no belief target is constructed.

## M16 · move/network

**Paper** (Table 24) — Encoder-only transformer, depth 8, embedding 384, 8 heads, feed-forward 1,536, 14.7M parameters, pre-layernorm, key-query product policy head, learned absolute positional embeddings.

**Phase 17** — The accepted C1: depth 4, width 128, 4 heads, feed-forward 512, 863,959 parameters, pre-layernorm, key-query policy head (policy_query/policy_key), learned_row_column_v1 position encoding, 127-channel observation, 10,000 action logits.

**Status** `scaled` · **owner** fixed upstream

**Reason** — Same family and same policy parameterization at ~1/17 the parameters -- the local scale decision made in Phase 6 and accepted since. Not revisited by Phase 17.

**Required test / telemetry** — Start-identity gate binds the architecture config and the 863,959 count.

## S01 · setup/architecture

**Paper** (Table 23; Fig. 3) — Decoder-only causal transformer over start token + 40 row-major piece tokens. Depth 4, embedding 512, 8 heads, feed-forward 2,048 (= 4 x width), learned positional embedding init std 0.1, pre-layernorm, 12.6M parameters.

**Phase 17** — Depth 4, width 128, 4 heads, feed-forward 512 (= 4 x width), pre-layernorm, ~0.80M parameters.

**Status** `scaled` · **owner** Agent 3

**Reason** — Width scaled 512 -> 128 with the paper's 4x feed-forward ratio preserved and depth kept at 4. ARITHMETIC CONFIRMING FF=512: 4 blocks x (attention 66,048 + feed-forward 131,712 + 2 layernorms 512) = 793,088, plus 13 token embeddings x 128 = 1,664, 41 positions x 128 = 5,248, final norm 256, and the three heads 1,548 + 387 + 129 -> 802,320 parameters, i.e. 'approximately 0.8 million'. Feed-forward width 51 would give 328,412 and could not reach 0.8M. 512 governs.

**Required test / telemetry** — Setup gate asserts the parameter count is within a declared band of 802,320 and that the config is 4/128/4/512.

## S02 · setup/causal factorization

**Paper** (D.3; Fig. 3) — Given a setup prefix in row-major order the network emits (1) W/L/D probabilities, (2) a real-valued conditional-entropy estimate, (3) a distribution over the next piece placement. Training on entire setups in single forward-backward passes.

**Phase 17** — At every one of the 40 prefixes: masked 12-way next-piece logits, W/D/L logits, and a scalar conditional-entropy prediction. Remaining inventory computed solely from the prefix; exhausted types receive an unsampleable mask.

**Status** `exact` · **owner** Agent 3

**Reason** — Identical factorization and identical head set.

**Required test / telemetry** — Setup gate: autoregressive causality test (prefix k's outputs are unchanged by tokens > k); exhausted-token adversarial masking test; zero inventory/legality/placement failures over >= 5,000 samples split across colours.

## S03 · setup/orientation

**Paper** (not in the paper (local engine concern)) — No analogue; the paper's simulator has one canonical frame.

**Phase 17** — Generate in canonical own-side coordinates; convert only at the engine boundary through the accepted Phase 15 helper (stratego.belief.phase15.orientation, rule 'red engine row == canonical rank; blue engine row == 9 - canonical rank', version phase15_orientation_rule_v1). Never generate Blue directly in engine orientation; never pass canonical Blue to create_game.

**Status** `not used` · **owner** Agent 3

**Reason** — Local-only requirement. It exists because the Phase 11B glue mis-oriented Blue setups: 77.0% Blue front-row flags under the old path versus 1.77% corrected.

**Required test / telemetry** — Setup gate: zero orientation failures; the accepted negative_canary and check_board are imported, never re-derived.

## S04 · setup/sampling and pools

**Paper** (D.2; Table 18) — A pool of pre-generated setups sampled from the setup network, 1,000 per player per GPU, REGENERATED AFTER EACH TRAINING ITERATION.

**Phase 17** — Vectorized fresh pools generated under each frozen raw setup snapshot; default 512-1,000 candidates per side per iteration; Agent 3 selects the smallest size that keeps game creation supplied without material training delay. Unused and refill counts recorded.

**Status** `exact` · **owner** Agent 3

**Reason** — Same regeneration cadence and same per-side pool scale; only the count is allowed to shrink to local throughput.

**Required test / telemetry** — Setup gate records generation cost; pool unused/refill counters in telemetry. A generation or orientation failure is FATAL -- there is no frozen setup library in Phase 17 training and no silent library fallback.

## S05 · setup/binding to a game

**Paper** (D.2; D.3) — A setup is drawn once at game creation from the pool; theta_t is 'the parameters that generated sigma-bar'; because games span iterations the setup data is slightly off-policy.

**Phase 17** — A setup is sampled once at game creation and stays bound to that game. Its behavior probabilities and setup-snapshot digest remain attached until the outcome arrives, even after the setup learner has updated. Deliberately UNLIKE the move side (row M03).

**Status** `exact` · **owner** Agent 3 / Agent 4

**Reason** — Identical, and the asymmetry with M03 is intentional: a setup is one decision made once, a move policy acts repeatedly.

**Required test / telemetry** — Setup episode records policy age and the generating snapshot digest; the ratio denominator is always the recorded behavior probability.

## S06 · setup/returns and filtering

**Paper** (2.3; D.3) — Monte Carlo returns -- the final outcomes of games played by the current policy -- with NO advantage filtering. All setups of games finished during the last data collection period are trained on.

**Phase 17** — Both sides train from the result: win +1, draw 0, loss -1 from that side's own perspective. All 40 prefixes, no move-style top-quartile filter.

**Status** `exact` · **owner** Agent 3

**Reason** — Identical.

**Required test / telemetry** — Setup gate: Red/Blue/draw outcome-sign tests and a synthetic reward-flip gradient test (flipping the outcome flips the gradient's sign).

## S07 · setup/advantage

**Paper** (D.3 (delta)) — delta = (o - E[v_theta_t(sigma)]) + alpha*(H(sigma-bar | sigma; theta_t) - h_theta_t(sigma)), with alpha the regularization temperature (the entropy-maximization coefficient).

**Phase 17** — Same two-term form. H is read as the REALIZED SUFFIX INFORMATION CONTENT I(sigma-bar|sigma;theta_t) = -log pi_theta_t(sigma-bar|sigma) in nats -- the only quantity computable from one sampled setup, and the Monte Carlo estimator of the conditional entropy. AMBIGUITY RESOLVED: the entropy term is frozen as alpha*(I/10 - h_theta_t(sigma)), i.e. BOTH sides in the normalized units the prediction loss trains.

**Status** `intentional divergence` · **owner** Agent 1 freezes / Agent 3 implements

**Reason** — As printed, Eq. (1) regresses h to H/10 while delta uses (H - h). Those are different units, and the mixed form degenerates to roughly 0.9*alpha*H -- an UNCENTERED bonus. With early setup entropy around 100 bits (~69 nats, paper Fig. 4B) that term is ~6.2 at alpha=0.1, which would swamp (o - E[v]) in [-2, 2]. The normalized reading keeps the entropy term commensurate with the outcome term and is the only reading under which alpha behaves as an advantage coefficient. Recorded as a deviation from the literal text and flagged for operator confirmation.

**Required test / telemetry** — Telemetry logs I, I/10, h and the two advantage terms separately so their relative magnitudes are visible from hour 0.

## S08 · setup/conditional-entropy head

**Paper** (Eq. 1; Table 20) — L_h = (H(sigma-bar|sigma;theta_t)/10 - h_theta(sigma))^2; conditional-entropy prediction loss coefficient 1; normalizing constant 1/10.

**Phase 17** — Identical, with H read as the realized suffix information content per row S07.

**Status** `exact` · **owner** Agent 3

**Reason** — Transcribed exactly, including the 1/10 normalizer and the coefficient of 1.

**Required test / telemetry** — Setup gate: h converges toward I/10 on the initial masked model; L_h logged separately.

## S09 · setup/value head

**Paper** (Eq. 2; Table 20) — L_v = -log v_theta(o | sigma); value loss coefficient 0.5. L_setup = L_pi + 0.5*L_v + L_h.

**Phase 17** — Identical: W/D/L cross-entropy at every prefix, weight 0.5, and the same total weighting.

**Status** `exact` · **owner** Agent 3

**Reason** — Transcribed exactly.

**Required test / telemetry** — Each of the three loss terms logged under its own name.

## S10 · setup/PPO clipping

**Paper** (Eq. 3; Table 20) — L_pi = -min(r*delta, clip(r, 0.8, 1.2)*delta), r = pi_theta(sigma+|sigma)/pi_theta_t(sigma+|sigma); importance ratio clipping parameter 0.2.

**Phase 17** — Identical: clipping 0.2, per-prefix ratio, behavior probabilities always from the recorded raw setup snapshot.

**Status** `exact` · **owner** Agent 3

**Reason** — Transcribed exactly.

**Required test / telemetry** — Setup clip fraction logged per epoch.

## S11 · setup/behavior KL

**Paper** (Eq. 3; Table 20) — + 0.1 * KL(pi_theta(sigma), pi_theta_t(sigma)) -- REVERSE KL (current || behavior) over the next-piece distribution, FIXED coefficient 0.1.

**Phase 17** — A separate adaptive setup behavior-KL controller, independent of the move controller. PROVISIONAL fields pending Agent 3 calibration: direction, target, beta0, beta bounds and hard range.

**Status** `intentional divergence` · **owner** Agent 1 schema / Agent 3 calibration

**Reason** — Fixed-vs-adaptive, and the direction must be stated rather than inherited. The move side's accepted controller measures the FORWARD KL, so 'the same controller' would silently flip the paper's direction. Agent 1 freezes the SCHEMA and marks the numbers provisional; Agent 3's soak supplies calibration; the operator confirms before launch.

**Required test / telemetry** — The direction is a required, logged field. Stop condition: setup KL above its hard range for three consecutive setup updates.

## S12 · setup/epochs

**Paper** (Table 20) — 5 epochs per training iteration, batches of 1,024 per GPU.

**Phase 17** — 5 setup epochs per setup iteration is the DEFAULT. Agent 3 may recommend fewer only with measured generation, forward/backward and total iteration costs showing that five materially threaten the 12-hour move budget. Silent reduction is prohibited.

**Status** `exact` · **owner** Agent 3

**Reason** — Transcribed exactly with an explicit, evidence-gated escape hatch.

**Required test / telemetry** — Setup gate: five-epoch setup throughput measurement, reported as generation / forward-backward / total per iteration.

## S13 · setup/gradient clip

**Paper** (D.3; Table 20) — Maximum gradient norm 0.5.

**Phase 17** — 0.5.

**Status** `exact` · **owner** Agent 3

**Reason** — Transcribed exactly. Note this differs from the move side's 1.0 (row M12) and the two must not be merged into one constant.

**Required test / telemetry** — Setup gradient norm logged separately from the move gradient norm.

## S14 · setup/learning rate

**Paper** (Table 20) — Adam, learning rate 5e-5, CONSTANT -- the paper schedules only the move learning rate (2.4).

**Phase 17** — Adam, 5e-5, constant.

**Status** `exact` · **owner** Agent 3

**Reason** — No horizon mapping is needed for a constant. Transcribed exactly.

**Required test / telemetry** — Setup scheduler position checkpointed even though the value is constant, so a resume is provably identical.

## S15 · setup/regularization temperature

**Paper** (Table 20; 2.4) — alpha(n) = 0.1 / n^0.3. Over N_paper ~= 42,376 this runs 0.100000 -> 0.004091, an anneal depth of 24.4x.

**Phase 17** — Endpoint-preserving re-horizon: alpha(n) = max(0.1 * n^(-p), 0.1 * N_paper^(-0.3)) with p = 0.3 * ln(N_paper)/ln(N) and N_paper = 42,376. Both endpoints are exact: alpha(1) = 0.100000 and alpha(N) = 0.004091 for any N. Example exponents: N=300 -> p=0.5604; N=626 -> p=0.4964.

**Status** `scaled` · **owner** Agent 1 freezes / Agent 4 substitutes N

**Reason** — Raw transcription onto a 12-hour horizon would END 3.5x more heavily regularized than the paper (N=626 gives alpha(N)=0.014489 against the paper's 0.004091) -- the setup policy would never leave the high-entropy regime. The exponent rescale is the minimal shape-preserving map: same power-law family, same first and last value, log axis rescaled. RISK, recorded not hidden: the paper warns that annealing too aggressively 'collapsed the entropy of the model'; traversing a week-long anneal in 12 hours is aggressive by construction. The floor makes the schedule safe on overrun. Flagged for operator confirmation.

**Required test / telemetry** — Instrumented by the existing stop policy: setup mean prefix entropy below 60% of its initial baseline for three checks, and flag effective support below four.

## S16 · setup/EMA

**Paper** (D.3; Table 20) — EMA of the setup parameters, smoothing 0.999, updated after the training iteration; used for evaluation.

**Phase 17** — Setup EMA decay 0.999, initialised from scratch with the setup model. RAW generates all setups; EMA is exported for evaluation only.

**Status** `exact` · **owner** Agent 3 / Agent 4

**Reason** — Transcribed exactly, with the same raw/EMA role split as the move side.

**Required test / telemetry** — Paired checkpoint stores raw and EMA setup states; the joint evaluation lane uses paired EMA weights.

## S17 · setup/episode supply

**Paper** (D.3) — 'All setups for games that were finished during the last data collection period were included in the training data.' Batches of 1,024 per GPU; 99.5e3 setup gradient steps over the run.

**Phase 17** — Completed setup episodes enter a bounded FIFO queue and are consumed once in a fixed setup-sequence budget. Queue depth, oldest/mean age, policy age, consumed count and any rejected or discarded episode are recorded. Silent dropping is prohibited. Too few episodes -> skip the setup update EXPLICITLY; repeated starvation is a production stop condition.

**Status** `scaled` · **owner** Agent 3 / Agent 4

**Reason** — Same 'finished since last collection' source with an explicit bounded queue, because a single machine finishes far fewer games per iteration than 16 GPUs and starvation is a real local failure mode rather than a theoretical one.

**Required test / telemetry** — Stop conditions: no setup optimizer update for one complete 30-minute interval after warm-up while games and setup episodes complete; setup queue age/backlog over the frozen ceiling for three windows.

## B01 · belief

**Paper** (2.5; D.5; Table 25; Fig. 7) — A separate 57.1M-parameter encoder+decoder belief network (encoder depth 6, 4 decoder blocks, 8 heads, width 512, dropout 0.2, feed-forward 2,048), trained AFTER the self-play run on trajectories of the FINAL policies, on 4 H100s for 4 days.

**Phase 17** — NOT BUILT IN PHASE 17. The move network's marginal belief head is present but zero-weighted (row M15). Joint autoregressive belief training is a later phase, permitted only after the operator promotes a Phase 17 checkpoint.

**Status** `not used` · **owner** none in Phase 17

**Reason** — Deliberate phase boundary. The paper's own ordering agrees: belief training is downstream of a finished policy.

**Required test / telemetry** — Structural assertion that no Phase 17 module constructs a belief world model.

## B02 · search

**Paper** (2.5; D.7; Table 26) — Test-time search: sample ~1,000/|legal| hidden configurations from the belief net, run 1,000 depth-40 rollouts, average value predictions, then take one tabular magnetic-mirror-descent step, pi_search proportional to exp(q/alpha) * rho^(...) * pi_theta^(...), alpha=0.002, beta=0.02, and SAMPLE the played move from it.

**Phase 17** — PROHIBITED in collection and in training. No Phase 17 agent implements or quietly prepares belief-guided search.

**Status** `not used` · **owner** none in Phase 17

**Reason** — Deliberate phase boundary, and it also matches the paper's own finding (2.3) that search-based data generation was not worth its cost.

**Required test / telemetry** — Correctness gate: structural no-search assertion over the collection path.

## B03 · game rules

**Paper** (D.1; Table 17) — Training under a 100-move battleless draw rule (evaluation under 200); maximum game length 4,000 moves. The proximity-to-rule observation channel is fractional, so n battleless moves read as n/100 in training and n/200 in testing.

**Phase 17** — The accepted local ruleset stratego_project_v1 is unchanged. Any change to the two-square, continuous-chasing, battleless or move-safety limits would require a NEW rules identifier (02_project_ruleset.md section 227) and is out of scope.

**Status** `not used` · **owner** fixed upstream

**Reason** — Changing the rules version would invalidate every accepted result and the whole evaluation stack. Not attempted.

**Required test / telemetry** — Correctness gate binds rules_version = stratego_project_v1 on every artifact.

## B04 · precision and hardware

**Paper** (2.6) — bfloat16 (roughly 3x faster self-play iterations), a CUDA C++ GPU-resident simulator at ~10e6 state updates/s, 16 H100s for one week.

**Phase 17** — float32 on Apple MPS, the accepted Phase 9 OPTIMIZER_CONSTRAINTS, on one M4 Pro for 12 hours.

**Status** `intentional divergence` · **owner** fixed upstream

**Reason** — Hardware reality. Recorded because it is the reason every population and horizon number in this map had to be re-derived rather than copied.

**Required test / telemetry** — Precision and device are checkpointed identity fields.

## B05 · training horizon

**Paper** (Table 27; D.2) — 163e6 finished games, 208e9 environment steps, 8.56e6 move gradient steps, 99.5e3 setup gradient steps. At 202 batches per move iteration that is N_paper = 8.56e6/202 ~= 42,376 training iterations.

**Phase 17** — One 12-hour run. N is MEASURED by Agent 4's bounded preflight throughput rehearsal and FROZEN, together with n_ref and the whole schedule curve, before h0. It is never recomputed from changing production speed.

**Status** `scaled` · **owner** Agent 1 derives N_paper / Agent 4 freezes N

**Reason** — N_paper = 42,376 is the derived constant every re-horizoning in this map depends on, so it is recorded once here rather than re-derived per row. The local N is roughly two orders of magnitude smaller, which is exactly why rows M07, M08 and S15 exist.

**Required test / telemetry** — The frozen N, n_ref, p and the full schedule table are bound into the launch record and re-verified on resume.
