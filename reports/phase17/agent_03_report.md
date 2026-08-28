# Phase 17 — Agent 3 follow-up report
## D7-B, D5, and the setup-diversity gate

**Status: gate G-S FAILED on S6. `ready_for_tandem_integration` is false.**

23 of 24 checks pass, 51.3 minutes, equation `phase17_setup_update_v2`, entropy bonus `0.9 * alpha * (I/10)`.

Both instructed changes were implemented exactly and both did what they were asked to do:
**D7-B works** — the entropy bonus is no longer a self-cancelling residual. **D5 is resolved** — the KL controller now genuinely regulates, after two defects in my own controller were found and fixed. Neither change stopped the setup distribution concentrating, which is what S6 measures and what still fails.

---

## 1. What three runs show

| | v1 | v2 | v3 (this run) |
|---|---|---|---|
| entropy term | `alpha*(I/10 - h)` centered (D4) | `0.9*alpha*(I/10)` (D7-B) | `0.9*alpha*(I/10)` (D7-B) |
| KL target | 0.015 | 0.0037 | **0.0018** |
| controller cadence | per epoch | per epoch | **per iteration, final-epoch KL** |
| **S6** entropy | FAIL (run 3) | pass (run 1) | **FAIL (run 4)** |
| **S12** controller | 100% pinned | FAIL, 98.9% pinned | **PASS, 14.5% at bound** |
| final mean prefix entropy | 0.8948 | 0.9003 | **0.8091** |
| final flag effective support | 5.53 | 6.97 | **7.63** |

All three end **below** the 0.9257 floor. v2's S6 pass was not a better outcome: it finished at essentially the same entropy as v1 (0.900 vs 0.895) and passed only because its two hard readings happened to fall non-consecutively. I want to be explicit that reading v2 as "D7-B fixed the entropy gate" would have been wrong, and v3 — a strictly better-behaved controller — makes that clear by failing harder.

### Entropy and flag support at every check

| iteration | v1 H | v2 H | v3 H | v1 flag | v2 flag | v3 flag |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 1.7054 | 1.7240 | 1.6655 | 10.77 | 11.90 | 10.67 |
| 100 | 1.4023 | 1.3377 | 1.5518 | 11.80 | 13.05 | 9.44 |
| 175 | 1.2736 | 1.3090 | 1.3737 | 9.93 | 6.52 | 8.22 |
| 250 | 1.3484 | 1.1230 | 1.1922 | 8.80 | 8.79 | 7.01 |
| 325 | 1.2491 | 1.0859 | 1.0341 | 8.42 | 5.51 | 9.73 |
| 400 | 1.0933 | 1.0974 | 0.9991 | 8.47 | 8.26 | 8.39 |
| 475 | 1.0885 | 1.0192 | 0.9175 | 7.07 | 7.56 | 7.77 |
| 550 | 0.9380 | 0.9160 | 0.9221 | 7.27 | 6.85 | 7.43 |
| 625 | 0.9067 | 0.8827 | 0.8056 | 5.53 | 6.97 | 8.03 |

Floor 0.9257 (60% of the 1.5429 baseline). v3 hard readings on cadence: [475, 500, 525, 550, 600, 625], longest consecutive run 4.

## 2. The concentration is real

Matched 320-sample comparison, soak iteration 1 against iteration 626 — so this is not an artifact of measuring a trained policy against a random baseline:

| metric | iteration 1 | iteration 626 | change |
|---|---:|---:|---:|
| mean prefix entropy (nats) | 1.5429 | 0.8091 | -48% |
| sequence information I_0 (nats) | 61.55 | 32.49 | -47% |
| mean per-square entropy (bits) | 2.8841 | 1.9522 | -32% |
| reflection-class uniqueness | 1.0000 | 1.0000 | +0% |
| mean class distance (of 40) | 31.50 | 25.09 | -20% |
| min class distance | 20 | 8 | -60% |
| flag effective support | 20.47 | 7.63 | -63% |
| flag square support | 28 | 12 | -57% |
| bomb effective support | 37.42 | 17.06 | -54% |
| bomb pattern uniqueness (of 320) | 320 | 167 | -48% |
| mean top-token concentration | 0.2780 | 0.4763 | +71% |

Every axis moved. Flag placement went from 28 squares to 12; the closest pair of setups went from 20 differing squares to 8. Only reflection-class uniqueness is unchanged, and at 320 samples that is a weak test. An earlier draft of this report described the distribution as "not collapsed in absolute terms" on the strength of the uniqueness and flag-floor numbers alone; the matched comparison does not support that and I have withdrawn it.

## 3. D7-B: implemented, and it did what it was meant to

```text
delta = (o - E[v]) + 0.9 * alpha * (I/10)
```

The conditional-entropy head and `L_h` (weight 1.0) are retained for telemetry and paper alignment; nothing in the advantage reads `h` any more. Five setup epochs kept.

Measured strength of the bonus relative to the outcome term:

| iteration | v1 (centered) | v2/v3 (D7-B) |
|---:|---:|---:|
| 1 | 1.2x weaker | 1.50x weaker |
| 10 | 24.9x weaker | 1.84x weaker |
| 400 | 106.0x weaker | 14.51x weaker |
| 626 | 107.2x weaker | 25.62x weaker |

So the residual-collapse mechanism is genuinely gone: the bonus is 4–13x stronger for most of the run. It is still 26x weaker than the outcome term by the end, because D3's anneal cuts alpha 24x over the horizon. Making the two comparable at iteration 626 would need alpha near 0.106 — essentially no anneal at all.

## 4. D5: resolved, after two defects in my own controller

Frozen: reverse KL, target **0.0018**, beta0 0.1, bounds [0.001, 1.0], hard limit 0.08.

Two defects had to be fixed before the operator's constants could even be judged:

1. **The controller stepped once per epoch.** Epoch 0 begins with the policy sitting exactly on the behavior snapshot, so its KL is near zero *by construction, not by evidence* — measured 0.000187, 0.000151, 0.000081 on the first three iterations against a decrease threshold of 0.00185. Every iteration handed the controller two tautological below-threshold readings and one above, netting a ~0.67x ratchet that drove beta to its floor in ~11 iterations and held it there for 98.9% of v2.

2. **It read the mean KL across epochs.** What a behavior-KL regulariser limits is where the policy *ends up* relative to the snapshot, not the average of the path. The mean understates the iteration's true drift by roughly 2.5x.

With both fixed, the target of 0.0037 still sat above the observed median (0.00176), giving 55% of iterations below the decrease threshold against 2.5% above — so beta still walked to its floor by iteration 30. That is what prompted the question that moved the target to the median.

Measured this run:

| quantity | value |
|---|---|
| beta range | 0.00100 – 0.3375 |
| iterations at lower bound | **14.5%** |
| iterations at upper bound | 0.0% |
| KL mean / p95 / max | 0.001721 / 0.003398 / 0.010403 |
| iterations over the hard limit | none |

The controller now regulates: beta spans more than two decades, never reaches its ceiling, and sits at its floor only 14.5% of the time. **S12 passes.**
## 5. The remaining criteria

| Criterion | Result |
|---|---|
| **S2** zero legality / inventory / orientation failures | PASS |
| **S3** exhausted-token masking | PASS |
| **S4** autoregressive causality | PASS |
| **S6c** reflection-class diversity healthy | PASS (min 1.0000) |
| **S6b** flag effective support ≥ 4 | PASS (min 6.68, final 7.63) |
| **S9** completed outcomes produce real setup updates | PASS |
| **S10** raw / EMA / optimizer / KL / queue round trip | PASS |
| **S11b** five setup epochs affordable | PASS (5.8% of a 12-hour run) |
| **S12** setup KL below hard limit, controller not pinned | PASS (14.5% at a bound) |
| **S6** no three consecutive entropy readings below 60% of baseline | **FAIL** (hard at [475, 500, 525, 550, 600, 625], run 4) |

## 6. Throughput — five epochs retained

| Measurement (CPU) | Value |
|---|---|
| pool of 1000/side | 3.05s (328/s) |
| pool of 512/side | 1.59s (322/s) |
| 1-epoch update, 320 episodes | 0.175s |
| 5-epoch update, 320 episodes | 0.815s |
| peak process memory | 2771 MiB |

Projected **3.99s per iteration**, **5.8% of a 12-hour run** (one epoch would be 4.9%). Unchanged from the previous gate; five epochs remain comfortably affordable. MPS timings are still unusable — they report 5 epochs as faster than 1 — so CPU is the figure used.

## 7. Corrections to my own earlier claims

- **"D3's anneal is not the cause."** True of v1, where the residual collapse dominated, but incomplete as a general claim. With the residual collapse removed by D7-B, the anneal is exposed underneath it: the bonus is still 26x weaker than the outcome term at iteration 626 purely because alpha has fallen 24x.

- **"A stronger beta should slow concentration."** I suggested this at iteration 200 of v3. The run went the other way. The controller penalises *reverse* KL, which is mode-seeking, so weighting it harder plausibly pushes the policy onto the behavior distribution's modes and compresses entropy. Offered as a candidate mechanism, not a finding — it rests on one comparison.

- **"The distribution is not collapsed in absolute terms."** Withdrawn. The matched-sample comparison in section 2 contradicts it on every axis.

- **A false claim of Phase 9 fidelity.** The setup KL controller's response ratios were documented as reusing "the accepted Phase 9 controller's multiplicative shape". They do not: Phase 9 increases above 2.0x its target and steps by 2.0 / 0.5; these are 1.5x and 1.5 / 1.5. The numbers are a legitimate choice — D5 left the ratios open — but the comment misrepresented their provenance and has been corrected.

## 8. What is still not established

- **Setup strength.** Unchanged from the previous report: the soak's move policy is a uniform-random legal fixture, so a better setup distribution has nothing to be better against.

- **Whether the concentration survives a real move signal.** 83% of the soak's games drew, so the outcome term is largely noise and a PPO policy fitting noise concentrates. This is the one factor common to all three runs and it is the confound I cannot remove within this agent's remit.

- **That D7-B is the best entropy form.** It is the one selected and it is measured here; no alternative coefficient or shape was swept.

- **That any alpha schedule fixes this.** The arithmetic in section 3 says what alpha would have to be; nothing was run to test it.

## 9. Operator decision requested — D9

**Setup-distribution concentration is not fixed by the entropy term or the KL controller.** Three recipe variants, same 626 iterations and seeds, all end below the floor.

| Option | Detail |
|---|---|
| **D9-A** treat D3's anneal as the lever | Making the bonus comparable to the outcome term at iteration 626 needs alpha ≈ 0.106 — essentially no anneal over 626 iterations. A large change to an accepted decision, and it needs its own arm. |
| **D9-B** defer S6 to a tandem soak *(recommended)* | The fixture confound is the largest unknown, and Agent 4's tandem runner produces exactly the soak that resolves it at no extra cost. Keep the absolute floors as production stop conditions meanwhile. |
| **D9-C** re-anchor S6 to absolute floors now | All of them pass. **Tension, stated plainly:** this is moving a floor after three runs tripped it — the move my own v1 report warned against. It should not be chosen because it produces a pass. |

I recommend **D9-B**. The concentration is real and should not be waved through, but three recipe variants failed to move it while the one factor common to all three — a noise-dominated outcome signal — is precisely what the tandem runner replaces.

## 10. Tests and artifacts

```bash
.venv/bin/python -m pytest tests/training/phase17 -q
```

**299 passed** (125 setup tests, 174 Agent 2's move/transition tests). The targeted setup selection alone is 125 passed. The full repository suite was not run, as instructed.

| Artifact | sha256 |
|---|---|
| `reports/phase17/phase17_setup_handoff_v1.json` | `07483067830f52bc2239364eafa3e2a57c664e7fae5eca5a4a31bea15a64f852` |
| `reports/phase17/agent_03_setup_gate.json` | `c77d87304c1de69b0b4b03af88ff8f1d09096f707b649f8c40c04f7e0bc933f1` |
| `reports/phase17/agent_03_setup_throughput.json` | `876b99e2584bc94a0eb79f5afa23677dc0b33e02fd111f10b2f0001cb27a969c` |

Source closure digest: `c0282a869a4634afb741b6b50b5827b744947ade35ad109932abc36094ce1438`

Config digest: `91ebf20447d07895f2d1a37caa3318ff2ad0072a666d722b56a0183c4be480de` · run `RUN-2026-A` · equation `phase17_setup_update_v2`

Determinism: the soak ran 626 iterations over 100,160 games and 15,650 optimizer steps, 0 starvation skips.
