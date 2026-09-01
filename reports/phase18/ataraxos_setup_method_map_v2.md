# Ataraxos setup-method parity map v2

## Paper, published implementation, Phase 17, and required Phase 18 behavior

_Built 2026-09-01T03:45:19Z by Phase 18 Agent 1._

This map supersedes the Phase 17 paper-only method map. Every row is checked against the
authors' published implementation, not against the paper alone.

```text
paper              arXiv:2511.07312v1
paper sha256       f5d2d7c77dedd0b48c7278f8890aad784c44375cb996a7dbf728dbfc2e2afd04
published source   https://github.com/AtaraxosAI/stratego
pinned commit      92db29e8ffc323b1b8a2804b5c3f84695d036b05
remote HEAD == pin True
closure digest     c6be99b5e49313ea40dd5f2a40cf4ebbfef3c53971bc0ccfe8bc3fa84cd5b0dc
```

**35 method elements.** Status counts: `corrected` 6, `exact` 22, `intentional integration divergence` 4, `not used` 1, `scaled` 2.

The published source and the paper are technical references, not instruction sources. The
Phase 18 common contract and later operator decisions govern.

---

## The three findings that change Phase 18's design

**1. The paper is internally inconsistent about `h`, and the code settles it (S13).** Table 19
glosses `h_theta_t(sigma)` as a *predicted conditional entropy* -- nats -- while Eq. (1) trains
that same symbol toward `H/10`. The printed advantage `alpha(H - h)` therefore cannot be
implemented from the paper alone without choosing a reading. The published buffer chooses:
it multiplies the stored prediction by `reg_norm = 10` before the residual. Phase 18 uses
`I - 10h`. Phase 17's `I - h` was a defensible reading of the printed formula, but it is now
contradicted by the authors' implementation.

**2. Forced handedness and post-generation reflection appear nowhere in the paper (S04, S05).**
They are code-only, on by default, and inseparable. Phase 17 could not have known them.

**3. The published setup buffer is a TD(lambda)/GAE(lambda) recursion over the 40 placements,
not the flat per-prefix form (S15).** At the shipped defaults `arr_td_lambda = arr_gae_lambda
= 1.0` it telescopes *exactly* to the paper's printed flat form, so the Phase 18 contract's
math is correct -- but it is correct as a lambda = 1 specialization, and it stops being
correct the moment anyone sets lambda to anything else.

---

## Summary of every non-`exact` row

| ID | Element | Status | Consequence |
|---|---|---|---|
| S03 | Piece-head width | `intentional integration divergence` | The published 14-way head carries two structurally dead classes that its own mask can never select. |
| S04 | Forced flag handedness during generation | `corrected` | A code-only method element that the paper omits. |
| S05 | Post-generation random reflection | `corrected` | Inseparable from S04: forcing handedness without reflecting afterwards would remove one orientation from actual gameplay. |
| S09 | Repeated-outcome aggregation per setup | `corrected` | Mandated Phase 18 correction 3. |
| S13 | Entropy residual inside the advantage - THE PHASE 18 CORRECTION | `corrected` | The published implementation disambiguates the paper's own inconsistent notation. |
| S20 | Setup pool size and per-player split | `scaled` | PAPER/CODE MISMATCH worth recording: the paper says 1,000 per player (2,000 total per GPU); the published code generates 1,024 total = 512 per player. |
| S26 | Batch size and epochs | `corrected` | Mandated Phase 18 correction 4. |
| S30 | Model-size scaling arithmetic | `scaled` | Proportional target = 12.6M * 0.863959M / 14.7M = 0.7405M. |
| S31 | Move-policy training regime | `intentional integration divergence` | This is the deliberate change of experimental point mandated by common contract 3.1 item 5. |
| S32 | On-policy requirement for setup PPO data | `intentional integration divergence` | The Phase 8 corpus provides move/value/belief supervision only. |
| S33 | Canonical/live mixture and setup-update cadence | `not used` | Nothing in the paper or the published code determines this, because they have no equivalent of the Phase 8 anchor stream. |
| S34 | Belief and search separation | `intentional integration divergence` | Common contract 3.1 item 5 restores the belief objective; the no-search and no-belief-architecture-replacement boundaries remain in force. |
| S35 | Signal handling and clean shutdown | `corrected` | A real published-code element that Phase 17 lacked. |

## Elements verified `exact`

| ID | Element |
|---|---|
| S01 | Setup sequence and causal factorization |
| S02 | Inventory masking |
| S06 | Behavior log-probability bookkeeping under reflection |
| S07 | Canonical-to-engine orientation |
| S08 | W/D/L category order and expected value |
| S10 | Setup identity and de-duplication |
| S11 | Realized suffix information I |
| S12 | Entropy-prediction normalization and target (Eq. 1) |
| S14 | Regularization temperature alpha(n) |
| S15 | TD(lambda)/GAE(lambda) recursion over the 40 placements |
| S16 | PPO ratio and clipping |
| S17 | Reverse behavior KL direction and coefficient |
| S18 | Loss weights |
| S19 | No advantage filtering for setups |
| S21 | Pool lifetime and buffer retention |
| S22 | Behavior snapshot binding |
| S23 | Aggregation window resets each collection period |
| S24 | Immediately terminal setups |
| S25 | Optimizer class and semantics |
| S27 | Gradient clipping |
| S28 | EMA and the raw/EMA actor split |
| S29 | Checkpoint, resume and file layout |

---

## Full map

### S01 - Setup sequence and causal factorization

**Status:** `exact`

**Paper.** D.3 opening paragraph: 'Given a setup prefix (meaning a setup partially filled in under row-major order), the setup network produces three outputs'. Table 19 defines sigma_bar, sigma, sigma+.

**Published code.** networks/arrangement_transformer.py:31-40,93-124 - decoder-only causal stack; docstring 'placing pieces from bottom to top, left to right'; forward() concatenates a start token then truncates to ARRANGEMENT_SIZE=40, so position t predicts placement t.

**Phase 17 behavior.** stratego/training/phase17/setup_model.py - 41-token sequence (start + 40), outputs read at prefixes 0..39; position 40 never read. Same factorization.

**Required Phase 18 behavior.** Keep the Phase 17 factorization. 41-token sequence, 40 readable prefixes, prefix k has seen exactly k placements.

**Reason.** Phase 17 already matches both paper and code. The two implementations differ only in whether the start token is prepended inside forward (code) or supplied by the caller (Phase 17); the alignment is identical.

**Required test.** Prefix-alignment test: for a known 40-token setup, assert output k is a function of tokens[0..k] only and is independent of tokens[k+1..]. Causality test: perturb token j>k and assert output k is bit-identical.

**Required telemetry.** none required

**Owning agent.** setup parity build agent

---

### S02 - Inventory masking

**Status:** `exact`

**Paper.** not printed

**Published code.** arrangement_transformer.py:127-160 - _remaining_counts subtracts the prefix cumsum from piece_counts; legal_mask = remaining > 0; illegal logits are filled with torch.finfo(dtype).min. Asserts at least one legal action at every prefix.

**Phase 17 behavior.** setup_sampling.batched_remaining / inventory_mask_from_prefix, derived from the prefix alone and never supplied by the caller.

**Required Phase 18 behavior.** Keep. Mask must be derived from the prefix, never passed in.

**Reason.** Identical semantics; both derive the mask rather than trusting a caller.

**Required test.** Exhausted-type test: after all 8 scouts are placed, assert P(scout)=0 exactly at every later prefix. Sum test: assert the 40 sampled tokens reproduce CLASSIC piece counts exactly.

**Required telemetry.** setup legality failures (must be 0)

**Owning agent.** setup parity build agent

---

### S03 - Piece-head width

**Status:** `intentional integration divergence`

**Paper.** not printed

**Published code.** utils/constants.py:24 N_PIECE_TYPE = pystratego.NUM_PIECE_TYPES; src/env/stratego_board.h:12 'const int NUM_PIECE_TYPES = 14; // Including empty and lake'. CLASSIC_INITIAL_COUNTS = {1,8,5,4,4,4,3,2,1,1,1,6,0,0}. The head is 14-way; lake and empty carry count 0 so the inventory mask makes them permanently illegal.

**Phase 17 behavior.** 12-way head (stratego/engine/constants.py NUM_PIECE_TYPES = 12); no lake/empty tokens exist.

**Required Phase 18 behavior.** Keep the 12-way head.

**Reason.** The published 14-way head carries two structurally dead classes that its own mask can never select. The 12-way head is functionally identical and is what the project's engine constants define. This changes the parameter count by 514 (token embedding 13 vs 15 rows, piece head 12 vs 14 outputs) and nothing else.

**Required test.** Assert the softmax over the 12-way head equals the published 14-way softmax restricted to the 12 live classes, on a canned prefix.

**Required telemetry.** none required

**Owning agent.** setup parity build agent

---

### S04 - Forced flag handedness during generation

**Status:** `corrected`

**Paper.** NOT PRINTED ANYWHERE. The paper does not mention handedness, symmetry reduction, or reflection.

**Published code.** arrangement_transformer.py:24 force_handedness: bool = True (default); :71-79 registers right_side = N_ARRANGEMENT_ROW * (5*[False] + 5*[True]) over the 40 square positions; :138-139 legal_mask[:, ~right_side, FLAG_IDX] = False. The flag is therefore illegal on the left five columns of every row. Docstring: 'Since equilibrium policies are left-right symmetric, the network supports forcing right-handedness ... to eliminate an axis of symmetry.'

**Phase 17 behavior.** ABSENT. Phase 17 placed no handedness constraint; both mirror images were learned as distinct sequences.

**Required Phase 18 behavior.** Force the Flag into one horizontal half during autoregressive generation, by masking FLAG_IDX at the squares of the other half.

**Reason.** A code-only method element that the paper omits. Phase 17 could not have known it from the paper; it is now visible and is one of the five mandated Phase 18 corrections (common contract 3.1 item 2).

**Required test.** Generate N setups with the mask on and assert the flag lands in the permitted half in 100% of them. Reduction test: with force_handedness disabled the module must reproduce the Phase 17 unconstrained distribution.

**Required telemetry.** flag column histogram; fraction of generated setups with the flag in the forced half (must be 1.0 before reflection)

**Owning agent.** setup parity build agent

---

### S05 - Post-generation random reflection

**Status:** `corrected`

**Paper.** not printed

**Published code.** arrangement/sampling.py:102-107 - after all 40 placements, flipped_mask = torch.rand(n_sample) > 0.5 and samples[flipped_mask] = flip_arrangements(...). arrangement/utils.py:15-29 flip_arrangements reshapes to (row, col, piece) and flips the COLUMN axis. The docstring at arrangement_transformer.py:38 warns: 'If force_handedness is True, the arrangement handedness should be randomized post-network generation.'

**Phase 17 behavior.** ABSENT (nothing to reflect, since generation was unconstrained).

**Required Phase 18 behavior.** Apply an independent seeded 50% horizontal reflection after generation. Record the reflection flag per setup.

**Reason.** Inseparable from S04: forcing handedness without reflecting afterwards would remove one orientation from actual gameplay. The two must ship together.

**Required test.** Reflection-parity test: assert reflect(reflect(x)) == x. Distribution test: over N draws assert the reflected fraction is 0.5 within binomial tolerance under a fixed seed. Play test: assert both flag halves occur in the setups actually handed to the engine.

**Required telemetry.** reflection flag rate; reflection-class support; flag column histogram AFTER reflection (must be symmetric)

**Owning agent.** setup parity build agent

---

### S06 - Behavior log-probability bookkeeping under reflection

**Status:** `exact`

**Paper.** not printed

**Published code.** arrangement/buffer.py:136-138 - the buffer stores arrangements in ENVIRONMENT orientation but log_probs in NETWORK orientation; :331-336 and :392 flip the arrangement back to network orientation BEFORE gathering the chosen-token NLL and before yielding training batches.

**Phase 17 behavior.** not applicable

**Required Phase 18 behavior.** Store the reflection flag with every setup and flip back to canonical/network orientation before any log-probability gather or PPO ratio. Never gather a log-probability against an engine-oriented board.

**Reason.** This is the single easiest place for S04/S05 to silently corrupt the ratio: a reflected board indexed against unreflected log-probs yields a wrong but finite NLL. The published code handles it explicitly and so must Phase 18.

**Required test.** Round-trip test: sample with reflection on, flip back, and assert the recovered token sequence equals the generated token sequence exactly, and that the recomputed suffix NLL equals the value recorded at generation to float tolerance.

**Required telemetry.** setup orientation failures (must be 0)

**Owning agent.** setup parity build agent

---

### S07 - Canonical-to-engine orientation

**Status:** `exact`

**Paper.** not printed

**Published code.** arrangement/utils.py:32-44 to_string -> pystratego.util.arrangement_strings_from_tensor; core/rl.py update_arrangements assigns even-indexed setups to red and odd-indexed to blue.

**Phase 17 behavior.** setup_sampling.to_engine_setup(canonical, player), orientation rule phase15_orientation_rule_v1: red engine row == canonical rank, blue engine row == 9 - canonical rank.

**Required Phase 18 behavior.** Convert only through the accepted boundary helper. Never hand a canonical setup to create_game directly.

**Reason.** The project already has an accepted, tested boundary helper. The Phase 11B/12 defect (77.0% Blue front-row flags) came from bypassing it, and that failure mode must not recur.

**Required test.** Orientation gate on every generated setup at every use; a failure is fatal, never a fallback. Assert blue and red conversions are mutual row reversals.

**Required telemetry.** orientation failures (must be 0); flag rank histogram by colour

**Owning agent.** setup parity build agent

---

### S08 - W/D/L category order and expected value

**Status:** `exact`

**Paper.** Table 19: o in {win=1, loss=-1, draw=0}; E[v_theta_t(sigma)] is the predicted expected outcome with win=1, loss=-1, draw=0.

**Published code.** utils/constants.py:97 CATEGORICAL_AGGREGATION = tensor([-1, 0, 1]) with the comment 'lose, tie, win', N_VF_CAT=3. buffer.py:260-261 maps scalar reward (-1,0,1) to index (0,1,2) via one_hot(reward+1). buffer.py:165-166 softmaxes the stored value logits. buffer.py:325 adv_est = adv_est @ CATEGORICAL_AGGREGATION.

**Phase 17 behavior.** expected_value_from_wdl(wdl) in setup_learning.py.

**Required Phase 18 behavior.** Index order is (loss, draw, win) = (0,1,2). E[v] = p_win - p_loss. Store value LOGITS at generation and softmax them when they enter the buffer, exactly as the published code does.

**Reason.** A silent index transposition here flips the sign of the entire outcome term and would look like a learning failure rather than a bug.

**Required test.** Canned test: a one-hot win must give E[v] = +1, a one-hot loss -1, a one-hot draw 0. Assert the aggregation vector is literally [-1,0,1] in that order.

**Required telemetry.** mean predicted E[v] by prefix; W/D/L reliability by prefix

**Owning agent.** setup parity build agent

---

### S09 - Repeated-outcome aggregation per setup

**Status:** `corrected`

**Paper.** D.3: 'We trained these outputs on all setups for games that were finished during the last data collection period'. D.2: the setup pool is 'regenerated after each training iteration', so one pooled setup can start several games. The paper prints Lv = -log v_theta(o|sigma) for a single o and does not print the aggregation.

**Published code.** buffer.py:264-271 - a RUNNING MEAN keyed by arrangement id: rewards[idx] = (counts[idx]*rewards[idx] + reward)/(counts[idx]+1); counts[idx] += 1; ready_flags[idx] = True. With use_cat_vf the averaged object is the one-hot W/D/L vector, so the value target becomes a SOFT distribution. buffer.py:190-199 reallocates counts/rewards/ready_flags to zeros on every add_arrangements call, so the aggregation window is exactly ONE collection period.

**Phase 17 behavior.** One freshly sampled setup per game, one outcome, immediately consumed. m = 1 always.

**Required Phase 18 behavior.** Aggregate every completed outcome attributed to the exact setup and behavior snapshot within the collection period. z_bar = mean z_i; the categorical target is the mean of the one-hot W/D/L outcomes. Record count and variance per setup. A setup with no completed game in the period is NOT trained and NOT treated as a draw.

**Reason.** Mandated Phase 18 correction 3. The published soft-target CE generalizes the paper's printed single-outcome Eq. (2): at m=1 they are identical, so this is a faithful reading of the code and not a departure from the paper.

**Required test.** Aggregation test: feed a known multiset of outcomes for one setup id and assert z_bar and the averaged one-hot match closed form. Ready-flag test: assert a setup with zero completed games is excluded from the batch entirely rather than contributing a draw.

**Required telemetry.** distribution of completed-outcome count m per setup (mean, median, min, max, fraction with m=1); outcome variance per setup; number of setups excluded for m=0

**Owning agent.** setup parity build agent

---

### S10 - Setup identity and de-duplication

**Status:** `exact`

**Paper.** not printed

**Published code.** buffer.py:176-187 - identity is pystratego PieceArrangementGenerator.arrangement_ids over the argmax board. Duplicates are collapsed by mark_most_recent_appearance, keeping the instance with the maximal step_added, so a re-sampled identical setup inherits the NEWER behavior snapshot and the older row is dropped. buffer.py:425-442 lookup_indices raises if a terminating game's arrangement is not in the buffer, so attribution is fail-loud.

**Phase 17 behavior.** canonical_fingerprint (content) and reflection_class_fingerprint (mirror class), used as diagnostics; no pool, so no de-duplication path existed.

**Required Phase 18 behavior.** Adopt an explicit setup identity. Two setups with the same canonical board sampled under different behavior snapshots must resolve to ONE trained row bound to the newer snapshot, matching the published semantics. Attribution failure must be fatal, never silently dropped.

**Reason.** Undocumented in the contract and easy to miss. Without de-duplication the same board can appear twice with different behavior log-probs and different outcome partitions, which double-counts it and splits its outcome evidence.

**Required test.** Duplicate test: add the same canonical setup under snapshots A then B and assert exactly one row survives, bound to B. Attribution test: a terminating game whose setup has been filtered must raise, not be discarded.

**Required telemetry.** duplicate collapse count per iteration; attribution failures (must be 0)

**Owning agent.** setup parity build agent

---

### S11 - Realized suffix information I

**Status:** `exact`

**Paper.** Table 19 defines H(sigma_bar | sigma; theta_t) as the conditional entropy of sigma_bar given sigma under theta_t. The paper does not print an estimator for it.

**Published code.** buffer.py:328-336 - nll = -log_probs gathered at the chosen token per prefix; the entropy recursion at :341-349 accumulates it backwards, which at td/gae lambda = 1 telescopes to sum_{j>=k} nll_j.

**Phase 17 behavior.** setup_sampling.suffix_information: I_k = reverse cumsum of -log pi_behavior(t_j | sigma_j), in nats. Identical quantity.

**Required Phase 18 behavior.** Keep. I_k = -sum_{j=k..39} log pi_behavior(t_j | sigma_j), in nats, computed under the BEHAVIOR snapshot.

**Reason.** I is a one-sample Monte Carlo estimate of H; both implementations agree.

**Required test.** Assert I_39 == -log pi(t_39) and I_k == I_{k+1} - log pi(t_k), for a canned log-prob array.

**Required telemetry.** I mean and quantiles by prefix

**Owning agent.** setup parity build agent

---

### S12 - Entropy-prediction normalization and target (Eq. 1)

**Status:** `exact`

**Paper.** Eq. (1): L_h = (H(sigma_bar|sigma;theta_t)/10 - h_theta(sigma))^2. So h is trained toward H/10.

**Published code.** buffer.py:357-359 reg_val_est = reg_val_est / reg_norm with arr_reg_norm = 10.0; rl.py:672 entropy_loss = F.mse_loss(regs_pred, batch.reg_returns). At arr_td_lambda = 1.0 the target reduces exactly to I_k / 10.

**Phase 17 behavior.** entropy_loss = ((normalized_information - conditional_entropy)^2).mean() with SETUP_CONDITIONAL_ENTROPY_NORMALIZER = 0.1, i.e. target = I/10.

**Required Phase 18 behavior.** Keep. target_h,k = I_k / 10; L_h = mean (h_theta,k - I_k/10)^2.

**Reason.** Phase 17 already matched the paper and the code on the TARGET. The defect was never in the loss; it was only in the advantage (S13).

**Required test.** Assert the fitted h converges to I/10 and not to I, on a canned constant-I fixture.

**Required telemetry.** mean h, mean I/10, residual (h - I/10)

**Owning agent.** setup parity build agent

---

### S13 - Entropy residual inside the advantage - THE PHASE 18 CORRECTION

**Status:** `corrected`

**Paper.** delta = (o - E[v_theta_t(sigma)]) + alpha(H(sigma_bar|sigma;theta_t) - h_theta_t(sigma)). Table 19 glosses h_theta_t(sigma) as 'Predicted conditional entropy of setup prefix sigma under theta_t' - i.e. in NATS - while Eq. (1) trains the same symbol toward H/10. The paper is internally inconsistent on this symbol and the printed formula alone cannot settle it.

**Published code.** buffer.py:303-305: ents = reg_norm * self.ents[self.ready_flags] with the explicit comment 'Multiplying by reg_norm gives network entropy prediction.' The residual at :343 is delta = nll[:,39] - ents[:,39] and at :347 delta = nll[:,k] + ents[:,k+1] - ents[:,k]. The stored h is therefore DENORMALIZED to 10h before it ever meets I.

**Phase 17 behavior.** setup_learning.setup_advantage line 215: entropy_term = alpha * (information - predicted), with `predicted` the RAW normalized h. This is I - h: raw nats minus a quantity normalized by ten.

**Required Phase 18 behavior.** Use I_k - 10*h_behavior,k. Equivalently H_hat_k = 10*h_k, residual = I_k - H_hat_k.

**Reason.** The published implementation disambiguates the paper's own inconsistent notation. Phase 17's I - h was a defensible reading of the printed formula when only the paper was available, but it is now contradicted by the authors' code and must not be relabelled a faithful transcription. Consequence of the defect: once h converges to I/10, I - h settles near 0.9*I, an uncentered POSITIVE bonus proportional to I, rather than the intended mean-zero innovation. Phase 17 shipped this knowingly (its own module docstring records it) after operator decision D10, and its printed advantage weighted the entropy term 2.70 against 1.00 for the outcome term.

**Required test.** Centering test: with h fitted to I/10, assert mean(I - 10h) is approximately 0 while mean(I - h) is approximately 0.9*mean(I). Canned test: for h = I/10 exactly, assert the residual is exactly 0.

**Required telemetry.** outcome term and entropy term separately: mean, absolute mean, quantiles, extrema, the entropy/outcome absolute-magnitude ratio, and each term's correlation with the total advantage

**Owning agent.** setup parity build agent

---

### S14 - Regularization temperature alpha(n)

**Status:** `exact`

**Paper.** Table 20: Regularization temperature = 0.1 / (Training iteration number)^0.3.

**Published code.** rl.py:756-760 power_schedule(coef, step, decay, ceil, floor) = min(max(coef/((step+1)^decay), floor), ceil); called at rl.py:466-472 with arr_temperature_coef=0.1, arr_temperature_decay=0.3, arr_temperature_ceil=1.0, arr_temperature_floor=0.001, and step = self.num_train_step, the GLOBAL training-iteration counter.

**Phase 17 behavior.** alpha(n) = 0.1 * n^-0.3 with n the shared one-based global tandem iteration. Matches; A4-CF6 was settled the same way.

**Required Phase 18 behavior.** alpha(n) = clip(0.1 / n^0.3, 0.001, 1.0) with n = global iteration index + 1 (one-based). Record the ceiling and floor even though neither binds.

**Reason.** The step+1 in power_schedule makes the paper's 'training iteration number' one-based. Neither clamp is reachable in a realistic run: 0.1*n^-0.3 <= 0.1 < 1.0 always, and the 0.001 floor is first reached at n = 100^(1/0.3) ~ 4.6e6 iterations. They are recorded so a future re-horizoning does not silently reintroduce them.

**Required test.** Assert alpha(1) == 0.1 and alpha(n) == 0.1*n^-0.3 over the planned iteration range, and that neither clamp activates within the planned horizon.

**Required telemetry.** alpha per iteration

**Owning agent.** setup parity build agent

---

### S15 - TD(lambda)/GAE(lambda) recursion over the 40 placements

**Status:** `exact`

**Paper.** NOT PRINTED for setups. The paper prints the flat form delta = (o - E[v]) + alpha(H - h). (It does print a lambda-return with lambda = 0.5 for MOVE learning, Table 21/D.4, which is a different objective.)

**Published code.** buffer.py:311-352 - the setup buffer runs a full backward TD(lambda) and GAE(lambda) recursion over the 40 placement steps for BOTH the outcome channel and the entropy channel, with arr_td_lambda and arr_gae_lambda. rl.py:76-77 sets both defaults to 1.0.

**Phase 17 behavior.** Flat form only. Phase 17 implemented the paper's printed per-prefix advantage directly and never had a lambda parameter, so it coincides with the published code at its defaults on this row.

**Required Phase 18 behavior.** Implement the flat form. At arr_td_lambda = arr_gae_lambda = 1.0 the published recursion telescopes exactly to the paper's printed flat form: the outcome GAE collapses to (z_bar - V_k) because there is no intermediate reward, and the entropy GAE collapses to I_k - 10h_k. The flat form is therefore not an approximation at the published defaults, it is algebraically identical.

**Reason.** This must be recorded explicitly, because the Phase 18 common contract states the flat form without noting it is the lambda=1 specialization of the published code. Anyone later tempted to set lambda != 1 must know that the flat form silently stops being correct at that point.

**Required test.** Equivalence test: implement the published recursion, run it at lambda=1, and assert it equals the flat form to float tolerance on random fixtures. Guard test: assert a lambda != 1 configuration is refused rather than silently computing the flat form.

**Required telemetry.** record td_lambda and gae_lambda in the config identity even though both are pinned to 1.0

**Owning agent.** setup parity build agent

---

### S16 - PPO ratio and clipping

**Status:** `exact`

**Paper.** Eq. (3): L_pi = -min(r*delta, clip(r,0.8,1.2)*delta), r = pi_theta(sigma+|sigma)/pi_theta_t(sigma+|sigma). Table 20: importance ratio clipping parameter 0.2.

**Published code.** rl.py:637-651 - log_prob and old_log_prob are both gathered at the CHOSEN token via batch.arrangements.argmax(-1); ratio = exp(log_prob - old_log_prob); policy_loss = -min(adv*ratio, adv*clamp(ratio, 1-0.2, 1+0.2)).mean(). arr_clip_range = 0.2.

**Phase 17 behavior.** Same form; setup_learning.setup_batch_loss line 357.

**Required Phase 18 behavior.** Keep. Ratio at the chosen token only, clip epsilon 0.2, mean over all prefixes and all rows.

**Reason.** Agrees across paper, code and Phase 17.

**Required test.** Canned test: r=1 must give L_PPO = -mean(delta); a delta>0 with r>1.2 must be clipped; a delta<0 with r<0.8 must be clipped.

**Required telemetry.** PPO ratio distribution; clip fraction; behavior-snapshot age

**Owning agent.** setup parity build agent

---

### S17 - Reverse behavior KL direction and coefficient

**Status:** `exact`

**Paper.** Eq. (3) adds 0.1*KL(pi_theta(sigma), pi_theta_t(sigma)); Table 20 names it 'Reverse KL to data collection policy loss coefficient 0.1'. Argument order is current first, behavior second.

**Published code.** rl.py:653 kl_loss = (log_probs.exp() * (log_probs - batch.log_probs)).sum(dim=-1).mean(), i.e. sum_a pi_current(a) * (log pi_current(a) - log pi_behavior(a)) = KL(current || behavior). arr_kl_coef = 0.1, fixed - there is NO adaptive KL controller for setups.

**Phase 17 behavior.** SETUP_KL_DIRECTION = 'reverse_current_given_behavior', fixed 0.1. Matches.

**Required Phase 18 behavior.** Keep. KL(pi_current || pi_behavior), fixed coefficient 0.1, no controller.

**Reason.** Direction agrees across all three. Note that the published setup KL relies on the illegal logits already being finfo.min so their exp() is ~0; the published MOVE KL by contrast multiplies explicitly by legal_actions. Phase 18 should mask explicitly rather than rely on that.

**Required test.** Direction test: assert KL(p||q) != KL(q||p) on an asymmetric fixture and that the implementation returns the former. Masking test: assert illegal actions contribute exactly 0 to the KL, not a small finite amount.

**Required telemetry.** mean reverse KL per iteration; per-prefix KL

**Owning agent.** setup parity build agent

---

### S18 - Loss weights

**Status:** `exact`

**Paper.** Eq. (4): L_setup = L_pi + (1/2)L_v + L_h, with the 0.1 KL already inside L_pi. Table 20: value 0.5, conditional entropy prediction 1, reverse KL 0.1.

**Published code.** rl.py:674-679 loss = arr_policy_coef*policy_loss + arr_ent_pred_coef*entropy_loss + arr_vf_coef*value_loss + arr_kl_coef*kl_loss, with 1.0, 1.0, 0.5, 0.1.

**Phase 17 behavior.** Same weights.

**Required Phase 18 behavior.** L_setup = L_PPO + 0.5*L_v + 1.0*L_h + 0.1*L_KL.

**Reason.** Unanimous.

**Required test.** Assert the four coefficients equal 1.0/0.5/1.0/0.1 in the frozen config identity.

**Required telemetry.** each loss component separately per iteration

**Owning agent.** setup parity build agent

---

### S19 - No advantage filtering for setups

**Status:** `exact`

**Paper.** not printed; no filter appears in D.3.

**Published code.** rl.py arr_train uses every row of every batch. The MOVE learner by contrast applies batch.adv_mask with adv_filt_rate 0.75 / adv_filt_thresh 0.01 (rl.py:63-64, 516-527). The setup learner has no counterpart.

**Phase 17 behavior.** No setup advantage filter.

**Required Phase 18 behavior.** Use all 40 prefixes of every eligible setup. Do not import the move-side advantage filter.

**Reason.** The asymmetry is deliberate in the published code and easy to copy across by accident.

**Required test.** Assert the setup batch row count equals 40 * (number of ready setups), with no mask applied.

**Required telemetry.** rows per setup update

**Owning agent.** setup parity build agent

---

### S20 - Setup pool size and per-player split

**Status:** `scaled`

**Paper.** D.2 Table 18: 'Generated setups per player 1,000 per GPU'; the pool is 'regenerated after each training iteration'.

**Published code.** rl.py:92 n_arr = 1024 TOTAL; update_arrangements generates 1024 and splits string_arrs[::2] to red, string_arrs[1::2] to blue, i.e. 512 per player.

**Phase 17 behavior.** No pool. One setup generated per game id, per colour.

**Required Phase 18 behavior.** One immutable pool of 1,024 canonical setups per snapshot, split 512 per player-use lane.

**Reason.** PAPER/CODE MISMATCH worth recording: the paper says 1,000 per player (2,000 total per GPU); the published code generates 1,024 total = 512 per player. The Phase 18 common contract follows the CODE. At a single-machine scale far below 16 GPUs the code's figure is the appropriate one, but the divergence is real and is not a transcription error on Phase 18's part.

**Required test.** Assert the pool has exactly 1,024 distinct entries and that the red and blue lanes receive 512 each with no overlap in assignment.

**Required telemetry.** pool size; per-lane counts; distinct canonical fingerprints per pool; reflection-class support

**Owning agent.** setup parity build agent

---

### S21 - Pool lifetime and buffer retention

**Status:** `exact`

**Paper.** D.3: 'Since most games span multiple reinforcement learning iterations, the training data was slightly off-policy.'

**Published code.** rl.py:174-178 ArrangementBuffer(storage_duration = N_PLAYER*train_every_per_player + env.conf.max_num_moves, ...) = 2*101 + max_num_moves. buffer.py:409-422 filter() drops rows whose step_added + storage_duration < current_step and sets need_arrangements. rl.py:284 filter is called after every arr_train. The retention window must cover the longest game plus the gap between pool refreshes, or add_rewards raises on lookup.

**Phase 17 behavior.** Not applicable; setups were consumed immediately.

**Required Phase 18 behavior.** Retain a setup at least (moves between pool refreshes) + (maximum game length) so that a game started under an old pool can still be attributed when it finishes. Sizing this too short is a hard error, not a silent loss.

**Reason.** This is the concrete mechanism behind the contract's 'slight setup off-policyness'. Getting it wrong produces a fatal lookup error rather than bad data, which is the desired failure mode.

**Required test.** Retention test: start a game under pool A, refresh to pool B, finish the game, and assert the outcome is attributed to A's row. Under-sized-window test: assert a too-short storage_duration raises rather than dropping the outcome.

**Required telemetry.** buffer row count; age distribution of trained setups; behavior-snapshot age in iterations

**Owning agent.** setup parity build agent

---

### S22 - Behavior snapshot binding

**Status:** `exact`

**Paper.** theta_t in Table 19 is 'Parameters that generated sigma_bar'.

**Published code.** add_arrangements stores values, ents and log_probs at generation time and they are never recomputed; process_data and arr_train read those stored tensors as the behavior quantities.

**Phase 17 behavior.** Every SampledSetup carries setup_model_state_digest and setup_snapshot_iteration; the advantage and the ratio denominator both belong to the snapshot that drew it.

**Required Phase 18 behavior.** Keep. The setup stays bound to the behavior snapshot that sampled it even if the raw model changes before its games finish.

**Reason.** Agrees; Phase 17 already did this correctly.

**Required test.** Assert the ratio denominator and the E[v]/h used in the advantage all come from the recorded snapshot, not from a re-forward of the current model.

**Required telemetry.** behavior snapshot digest and iteration per setup; snapshot age distribution

**Owning agent.** setup parity build agent

---

### S23 - Aggregation window resets each collection period

**Status:** `exact`

**Paper.** D.3: trained on 'all setups for games that were finished during the last data collection period'.

**Published code.** buffer.py:190-199 - counts, rewards and ready_flags are REALLOCATED TO ZERO inside every add_arrangements call, which happens once per training iteration. A setup surviving several periods therefore restarts its outcome count each period.

**Phase 17 behavior.** Not applicable.

**Required Phase 18 behavior.** Aggregate within one collection period only. A setup that persists across periods does not accumulate outcomes across them.

**Reason.** Not stated in the Phase 18 common contract and easy to implement the other way. Accumulating across periods would mix outcomes from different opponent populations and would inflate m without adding independent information about the current opponent distribution.

**Required test.** Assert counts and ready_flags are zeroed at each pool refresh and that a setup surviving two periods trains on the second period's outcomes only.

**Required telemetry.** m distribution per period; number of surviving setups whose counts reset

**Owning agent.** setup parity build agent

---

### S24 - Immediately terminal setups

**Status:** `exact`

**Paper.** not printed

**Published code.** rl.py update_arrangements: string_arrs = filter_terminal(to_string(tensor_arrs)) removes setups that are terminal at ply 0 from the ENVIRONMENT pool, but they remain in the buffer with ready_flags False and are simply never trained.

**Phase 17 behavior.** not applicable

**Required Phase 18 behavior.** Filter immediately terminal setups from play. Do not train them, and do not treat their absence as a draw.

**Reason.** Consistent with the m=0 exclusion rule in S09.

**Required test.** Assert a terminal setup is excluded from play and contributes no training row.

**Required telemetry.** terminal-setup rejection count per pool

**Owning agent.** setup parity build agent

---

### S25 - Optimizer class and semantics

**Status:** `exact`

**Paper.** D.3: 'using Adam with a learning rate of 5e-5'.

**Published code.** core/train_container.py:22-34 - torch.optim.AdamW with two param groups: non-bias params get weight_decay=cfg.arr_weight_decay, bias params get 0.0. rl.py:88 arr_weight_decay = 0.0. Betas and eps are PyTorch defaults (0.9, 0.999), eps 1e-8.

**Phase 17 behavior.** Adam-compatible constant-LR setup optimizer, lr 5e-5.

**Required Phase 18 behavior.** Use torch.optim.AdamW with weight_decay 0.0 on every group, betas (0.9, 0.999), eps 1e-8, constant lr 5e-5. Record the class explicitly.

**Reason.** RESOLVED, not an open method difference: AdamW with weight_decay=0 performs exactly the same update as Adam with weight_decay=0, because the decoupled decay term is multiplied by zero. The paper's 'Adam' and the container's 'AdamW(wd=0)' are the same optimizer here.

**Required test.** Parity test: run 100 steps of Adam(wd=0) and AdamW(wd=0) from the same init on the same batches and assert the parameters agree to float tolerance. Guard test: assert the frozen config refuses a nonzero setup weight decay.

**Required telemetry.** lr per iteration (constant); optimizer class in the config identity

**Owning agent.** setup parity build agent

---

### S26 - Batch size and epochs

**Status:** `corrected`

**Paper.** D.3: '5 epochs over the setups associated to games that completed over the previous data collection in batches of 1,024'. Table 20: batch size 1,024 per GPU, 5 epochs.

**Published code.** rl.py:625-632 - arr_batch_size = 1024, arr_num_epoch_per_train = 5; num_batches = ceil(ready_examples / 1024); an optimizer STEP is taken per 1,024-setup minibatch, with no gradient accumulation.

**Phase 17 behavior.** minibatch_episodes = 32 (setup_contract.py:370), 5 epochs.

**Required Phase 18 behavior.** Effective batch 1,024 setup episodes, 5 epochs. Gradient accumulation is permitted ONLY if device memory forces it, and only in the exact-equivalence sense of accumulating to one 1,024-row step.

**Reason.** Mandated Phase 18 correction 4. Note precisely what 1,024 means: it is the OPTIMIZER MINIBATCH in the published code, not an accumulated effective batch. Phase 17 took 32 optimizer steps where the published recipe takes one, so Phase 17's setup policy moved roughly 32x more often per unit of data and at a far noisier gradient. At width 128 the 1,024x40 activation footprint is small, so accumulation is very unlikely to be needed on MPS.

**Required test.** Assert one optimizer step per 1,024 ready setups per epoch, and 5 epochs per setup update. If accumulation is used, assert the accumulated gradient equals the single-step gradient to float tolerance.

**Required telemetry.** ready-setup count, minibatch count, optimizer steps per iteration

**Owning agent.** setup parity build agent

---

### S27 - Gradient clipping

**Status:** `exact`

**Paper.** D.3: 'We clipped the gradient norm for this loss at 0.5.' Table 20: maximum gradient norm 0.5.

**Published code.** rl.py:683-686 clip_grad_norm_(arr params, arr_max_grad_norm=0.5) before optim.step().

**Phase 17 behavior.** SETUP_GRADIENT_CLIP_NORM = 0.5.

**Required Phase 18 behavior.** Keep 0.5, applied to the setup parameters only, before the step.

**Reason.** Unanimous.

**Required test.** Assert the post-clip global norm never exceeds 0.5 and that the clip is applied to setup parameters only.

**Required telemetry.** pre-clip gradient norm distribution; clip activation rate

**Owning agent.** setup parity build agent

---

### S28 - EMA and the raw/EMA actor split

**Status:** `exact`

**Paper.** D.3: 'We updated an exponential moving average of the parameters - which was used for evaluation - after completion of the training iteration with a smoothing parameter of 0.999.'

**Published code.** networks/exponential_weighted_average.py:21-26 - ema.mul_(decay).add_((1-decay)*orig) over model.parameters() only, NOT buffers. rl.py:688 self.arr_ema_policy.update() is called ONCE per arr_train, after all 5 epochs and all minibatches. Generation uses self.arrangement_actor = the RAW net (rl.py:169); the EMA model is used only in save() and perform_evaluation().

**Phase 17 behavior.** SETUP_EMA_DECAY = 0.999; EMA updated per iteration; raw generates, EMA evaluates.

**Required Phase 18 behavior.** EMA decay 0.999, updated once after each complete setup update (not per minibatch). Raw weights generate pools and are trained; EMA weights are used only for validation, candidate export and final evaluation and never enter the training population.

**Reason.** Agrees. Two details worth carrying: the EMA covers parameters only, which is safe here because every buffer in the setup model is a constant; and the once-per-iteration cadence matters, because a per-minibatch EMA at 0.999 would track the raw weights far more closely and change what 'EMA' means.

**Required test.** Cadence test: assert exactly one EMA update per setup iteration regardless of minibatch count. Closed-form test: after k updates from a fixed raw model, assert the EMA equals the analytic geometric blend. Isolation test: assert the EMA model is never selected as the generation actor.

**Required telemetry.** EMA/raw divergence per iteration; digests of both

**Owning agent.** setup parity build agent

---

### S29 - Checkpoint, resume and file layout

**Status:** `exact`

**Paper.** not printed

**Published code.** train_container.py:19-38 loads weights from a checkpoint and the optimizer state from the same path with .pthw -> .ptho; EMA loads .pthw -> .pthm. rl.py:167 derives the setup paths by cfg.resume_from.replace('/model','/arr_model'), so the setup model, its optimizer and its EMA are three separate files with their own identities.

**Phase 17 behavior.** Phase 17 carried a real resume defect: the setup EMA was restored onto the payload's device, so any resumed MPS run died on the first setup update. Fixed at 3be8bba. A CPU-only resume proof cannot exercise it.

**Required Phase 18 behavior.** Persist raw weights, setup optimizer state and EMA weights as three separately identified objects. The resume proof must run on the production device, not only on CPU.

**Reason.** The Phase 17 defect is a device-placement bug that a CPU resume proof structurally cannot catch. Phase 18 must not inherit the CPU-only proof.

**Required test.** Round-trip test on the PRODUCTION device: save, reload, and assert raw, optimizer and EMA states all restore and that one further update runs without error.

**Required telemetry.** digests of raw, optimizer and EMA at every checkpoint

**Owning agent.** setup parity build agent

---

### S30 - Model-size scaling arithmetic

**Status:** `scaled`

**Paper.** Table 23 setup network: depth 4, embedding 512, 8 heads, feedforward 2,048, learned positional init std 0.1, total 12.6 million. Table 24 move network: depth 8, embedding 384, 8 heads, feedforward 1,536, total 14.7 million.

**Published code.** ArrangementTransformerConfig defaults depth=4, n_head=8, embed_dim_per_head_over8=8 -> embed_dim = 8*8*8 = 512; SelfAttentionLayer ff_factor=4 -> 2,048; pos_emb_std 0.1. Counting these defaults with N_PIECE_TYPE=14 and N_VF_CAT=3 gives 12,647,954 trainable parameters, which is Table 23's 12.6 million EXACTLY. MoveTransformerConfig defaults depth=8, embed_dim = 8*6*8 = 384, n_head=8, ff_factor=4 -> 1,536, matching Table 24's shape.

**Phase 17 behavior.** Phase17SetupModel at 4 blocks / width 128 / 4 heads / feedforward 512, 802,320 trainable parameters, counted in-process.

**Required Phase 18 behavior.** Keep 4/128/4/512 at 802,320 parameters as the governing default.

**Reason.** Proportional target = 12.6M * 0.863959M / 14.7M = 0.7405M. The Phase 17 architecture is 802,320, which is +8.35% above that target and preserves the paper's depth (4) and its 4x feedforward ratio. The scaling numerator is now VERIFIED from code rather than taken from the paper; the denominator (14.7M) remains paper-stated because the move network's parameter count depends on feature-orchestration components that need the compiled CUDA extension to instantiate. No architecture sweep is authorized before a demonstrated capacity or throughput failure.

**Required test.** Assert the built model has exactly 802,320 trainable parameters (SETUP_PARAMETER_TOLERANCE is 0). Assert positional embeddings initialize at std 0.1.

**Required telemetry.** parameter count and architecture digest in every artifact

**Owning agent.** setup parity build agent

---

### S31 - Move-policy training regime

**Status:** `intentional integration divergence`

**Paper.** D.4/Table 21-22: the move policy is trained by CURRENT-POLICY SELF-PLAY PPO with a lambda=0.5 return, an adaptive KL to a magnet policy, an advantage filter, and its own lr schedule.

**Published code.** rl.py train() - the move learner is PPO over self-play rollouts.

**Phase 17 behavior.** Tandem current-policy self-play, i.e. the paper's regime. Result: 24 of 24 trained move candidates scored below the hour-0 start, and the hour 6-12 move-only slope was negative (t = -2.97).

**Required Phase 18 behavior.** Phase 18 does NOT do self-play move learning. The move/value/belief heads are trained by the accepted Phase 8 SUPERVISED teacher objective on the canonical corpus plus live-stream trajectories.

**Reason.** This is the deliberate change of experimental point mandated by common contract 3.1 item 5. It also means the paper's move-side hyperparameters (lambda 0.5, magnet KL, advantage filter, lr schedule) are NOT USED and must not be imported. The setup objective is unchanged by this: it consumes game outcomes, and it does not care how the move policy was produced.

**Required test.** Assert no Phase 9 self-play or RL path is reachable from the Phase 18 trainer; the existing no_phase9_selfplay_or_rl gate covers this.

**Required telemetry.** none required

**Owning agent.** tandem pilot agent

---

### S32 - On-policy requirement for setup PPO data

**Status:** `intentional integration divergence`

**Paper.** The setup policy is trained only on setups it sampled.

**Published code.** Only pooled setups generated by the raw setup model ever enter the arrangement buffer.

**Phase 17 behavior.** Same.

**Required Phase 18 behavior.** Only the LIVE stream supplies setup PPO data. The accepted Phase 8 corpus was generated by the frozen neutral_v1 sampler and must never be treated as though the learned setup policy drew it.

**Reason.** The Phase 8 corpus provides move/value/belief supervision only. Using its neutral_v1 setups as setup-PPO rows would be invalid off-policy training with no recorded behavior distribution to form a ratio against - there is no pi_behavior for them at all.

**Required test.** Assert every setup-PPO row carries a behavior snapshot digest belonging to a Phase 18 setup model, and that no corpus-derived setup can reach the setup optimizer.

**Required telemetry.** count of setup rows by provenance (must be 100% live)

**Owning agent.** tandem pilot agent

---

### S33 - Canonical/live mixture and setup-update cadence

**Status:** `not used`

**Paper.** not applicable - the paper has no supervised anchor stream.

**Published code.** not applicable

**Phase 17 behavior.** not applicable

**Required Phase 18 behavior.** PROVISIONAL. The canonical/live example mixture and the setup-update cadence are research variables, to be frozen only after a bounded, predeclared tandem pilot.

**Reason.** Nothing in the paper or the published code determines this, because they have no equivalent of the Phase 8 anchor stream. It cannot be derived and must not be selected on a full run.

**Required test.** none at parity stage

**Required telemetry.** mixture ratio and cadence in the config identity once frozen

**Owning agent.** tandem pilot agent

---

### S34 - Belief and search separation

**Status:** `intentional integration divergence`

**Paper.** Belief is a separate 57.1M network (Table 25); test-time search is Section 4 / Table 26.

**Published code.** pyengine/belief/*, pyengine/core/search.py - both separate from the arrangement path.

**Phase 17 behavior.** Belief learning was DISABLED in Phase 17.

**Required Phase 18 behavior.** Belief supervision is RE-ENABLED as one of the three Phase 8 heads (lambda_belief 1.0). The paper's 57.1M belief architecture is NOT adopted, and no decision-time search is added.

**Reason.** Common contract 3.1 item 5 restores the belief objective; the no-search and no-belief-architecture-replacement boundaries remain in force. Nothing in the setup objective touches belief.

**Required test.** The existing no_decision_time_search gate, plus a belief-head learning gate on the sealed Phase 8 test.

**Required telemetry.** belief CE ratio and top-1; belief calibration by ply/reveal bucket on unusual setups

**Owning agent.** tandem pilot agent

---

### S35 - Signal handling and clean shutdown

**Status:** `corrected`

**Paper.** not printed

**Published code.** rl.py:205 signal.signal(signal.SIGUSR1, self.handle_sigusr1); should_terminate() checks the flag and the loop exits through save/evaluate/log before returning 'JOB COMPLETED SUCCESSFULLY'.

**Phase 17 behavior.** The frozen Phase 17 closure had NO signal handling, so SIGTERM killed the run instantly and session.close() never ran.

**Required Phase 18 behavior.** Adopt a cooperative termination signal so a stop lands at an iteration boundary with a saved checkpoint.

**Reason.** A real published-code element that Phase 17 lacked. It is not a learning-method item, but it is the difference between a clean stop and a lost iteration, and Phase 17's operator termination is exactly the situation it protects.

**Required test.** Send the signal mid-iteration and assert the run completes the iteration, saves, and exits zero.

**Required telemetry.** termination reason recorded in the closeout

**Owning agent.** production rehearsal agent

---

## Required derivations

### Entropy units (common contract 5.1)

Paper Eq. (1) trains

```text
L_h = ( H(sigma_bar | sigma; theta_t)/10  -  h_theta(sigma) )^2
```

so `h` predicts `H/10`. The published buffer then executes, at `arrangement/buffer.py:303-305`:

```text
ents = reg_norm * self.ents[self.ready_flags]   # reg_norm = arr_reg_norm = 10.0
```

with the in-source comment *Multiplying by reg_norm gives network entropy prediction*, and
forms the residual against the realized suffix NLL at lines 341-349. At `arr_gae_lambda = 1.0`
that recursion telescopes to

```text
reg_gae_trace_k  =  sum_{j=k..39} nll_j  -  10 * h_k  =  I_k - 10 h_k
```

Therefore the Phase 18 sampled-path estimator is `I - 10h`.

The practical difference is not cosmetic. Once `h` has converged to `I/10`:

```text
I - 10h  ->  0            a centered, mean-zero innovation
I -   h  ->  0.9 * I      an uncentered positive bonus proportional to I
```

Phase 17 shipped the second form. Its own module docstring records that the entropy term was
consequently weighted 2.70 against 1.00 for the outcome term in the printed advantage.

### Pool and reward flow (common contract 5.2)

```text
paper / published code:
  sample pool of 1,024 setups under ONE frozen behavior snapshot
    -> many game resets draw from that pool
    -> every completed game's outcome is folded into a running mean keyed by setup identity
    -> after the collection period, each setup with >= 1 completed game trains on z_bar

Phase 17:
  sample ONE fresh setup per game id
    -> exactly one outcome
    -> immediate per-episode PPO, m = 1 always
```

The published running mean is `buffer.py:264-271`:

```text
rewards[idx] = (counts[idx] * rewards[idx] + reward) / (counts[idx] + 1)
counts[idx]  += 1
ready_flags[idx] = True
```

With `use_cat_vf` the averaged object is the one-hot W/D/L vector, so the value target becomes
a soft distribution -- a generalization of the paper's printed single-outcome `L_v = -log v(o|sigma)`
that coincides with it at `m = 1`.

Phase 18 setup identity and attribution must therefore carry, at minimum:

```text
canonical setup fingerprint          reflection-class fingerprint
behavior setup-model digest          behavior snapshot iteration
pool id                              pool generation iteration
canonical and engine orientation     reflection flag
colour and setup-owner perspective   contributing game ids
completed outcome count m            outcome mean z_bar and variance
```

Two further semantics are required and are **not** stated in the common contract:

- **de-duplication (S10)**: identical canonical setups collapse to one row bound to the newest
  behavior snapshot; and

- **window reset (S23)**: `counts`, `rewards` and `ready_flags` are zeroed at every pool refresh,
  so aggregation spans exactly one collection period and never accumulates across periods.
