# Phase 16 — Agent 4
## Joint autoregressive belief: coherent worlds, then the ladder rerun

## Mission

The deep-search pilot proved the payoff for deeper search exists (+0.042 EWR at
LARGE for the oracle) and that the current world distribution is what fails
(−0.075 for belief-sampled worlds at the same budgets, worst-opponent decaying
0.833→0.667→0.500). The mixture pilot proved no reweighting of the *marginals* can
fix it: the regression is invisible at root-decision level (110/120 positions tied).
The failing property is that every sampled world is drawn from one shared marginal
field — wrong in the same way.

Build a **joint autoregressive belief** (the paper's Figure-7 shape at small scale):
decode the hidden army sequentially, each piece conditioned on the pieces already
decoded, so sampled worlds are coherent and diverse by construction. Then rerun the
deep ladder under its gate.

Do NOT scale the marginal head — that path is measured dead (overview §7).

Read `00_PHASE_16_OVERVIEW.md` first; its rules bind.

## 1. Process boundary and namespaces

Boundary identical to Agent 1 section 1. Training here is CPU-scale (minutes per
epoch) and needs no compute lock; the section-6 ladder pack is heavy compute and
does. Namespaces:

```text
stratego/belief/phase16/         tests/belief/phase16/
scripts/run_phase16_agent04.py   checkpoints/phase16/
reports/phase16/
```

## 2. Data — reuse, never regenerate first

`phase15_belief_corpus_v1`, read-only, digest-verified
(`b0493a08d2fcb1dd…`): 155,027 orientation-correct positions, per-family split
discipline already enforced. Reuse the Phase 15 feature pipeline and its frozen-P24
prefix caches by import (`ensure_caches` rebuilds the ~15 GB if absent — CPU, allow
the time once). Train on `train`, tune on `calibration`, report on `development` —
the same splits, so numbers are comparable with the Phase 15 tables.

## 3. The model

- **Decode order**: the corpus's stored hidden-piece order (row-major over the
  observer's hidden squares) — fixed and documented.
- **Conditioning per step**: the frozen-prefix features for the square (Phase 15
  pipeline), an embedding of every previously decoded rank with its square, and an
  explicit remaining-inventory count vector.
- **Architecture default**: 2 transformer decoder blocks, d_model 128, 4 heads,
  GELU, dropout 0.1 (config; the paper uses 0.2 specifically for out-of-distribution
  opponents — measure both at inference). A GRU fallback is acceptable if it beats
  the transformer on development NLL; report both if built.
- **Loss**: stepwise cross-entropy = joint NLL by the chain rule.
- **Masks in training AND sampling**, imported from the accepted constraint stack
  (never reimplemented): inventory exhaustion, moved-piece immobility (no Flag/Bomb
  on moved pieces), public admissibility. Ancestral sampling under these masks
  yields legal complete armies by construction; every sampled world must still pass
  the accepted validation stack (same check Phase 15 ran).
- Trainer discipline per Phase 15 Agent 1: CPU, AdamW, cosine, early-stop on
  development NLL, gradient isolation test (0 policy/value parameters receive
  gradients), digest-bound checkpoint that refuses a wrong backbone.

## 4. Predeclared metrics and gates (measure before integrating anything)

On the same 20,013-position development split, against B24 and the
`remaining_count` baseline:

1. **Joint NLL per hidden piece** — chain rule makes it comparable to B24's
   marginal CE (1.9709). Gate to proceed: **AR ≤ 1.9609** (−0.01).
2. **World-set metrics** over K = 32 sampled worlds on 512 paired development
   positions, arms: AR-sampler, B24+accepted sampler, count+accepted sampler:
   - mean per-piece agreement with the true army;
   - mean pairwise disagreement between sampled worlds (diversity);
   - true-flag coverage: fraction of positions where ≥1 world places the flag on
     its true square;
   - wall-clock per 32-world draw (target p95 ≤ 50 ms; batch the K worlds — the
     sequential dimension is ≤ 40 tiny steps).
   Gate to proceed: AR ≥ B24-sampler on **agreement AND diversity AND flag
   coverage** simultaneously, paired. (The marginal sampler trades these off;
   the joint model's claim is raising them together. If it does not, iterate once,
   then park and report — do not force it.)
3. Keep the Phase 15 breakdown axes (color, opponent class, game band) for the
   NLL table.

## 5. The provider

Implement the Phase 15 provider interface exactly (`predict_marginals`,
`sample_worlds`) in `stratego/belief/phase16/provider.py` so the frozen search
consumes AR worlds with zero engine changes. `predict_marginals` may return
sample-frequency marginals or single-pass step marginals — document which.
Reproduce the Phase 15 interface checks: determinism from seed, legality of every
world, truth isolation (`uses_hidden_truth = False`, public document only),
marginal latency.

## 6. The ladder rerun (Wave 2 — compute lock; ~10 h pack)

Reuse the Phase 15 deep-pilot machinery by import; identical design — 60 balanced
paired boards, same seeds, same opponents:

```text
arms   MEDIUM(B24)   MEDIUM(AR)   LARGE(AR)   + oracle at MEDIUM and LARGE
```

Predeclared decisions:

```text
worlds_fixed     LARGE(AR) - MEDIUM(AR) >= 0.00  (the sign flips)
no_regression    MEDIUM(AR) >= MEDIUM(B24) - 0.03
promote_deeper   both above hold AND LARGE(AR) >= MEDIUM(AR) + 0.03
                 -> maximum_strength may move to LARGE(AR); caps re-measured idle,
                    p95 must fit the 5.0 s ceiling (B24@LARGE measured 3.918 s —
                    AR sampling must not push past 5.0)
still_binding    worlds_fixed fails -> world quality is still the constraint;
                 report which metric in section 4 failed to predict it, and stop.
```

The ladder stays closed unless `worlds_fixed` passes. XLARGE stays retired
regardless (latency ceiling).

## 7. Candidate freeze and handoff

`checkpoints/phase16/phase16_belief_ar_candidate_v1.json` binding the AR checkpoint
digest, backbone digest, decode order, mask stack version, section-4 numbers, and
the section-6 verdict — or, if parked, the same file with `promoted: false` and the
measured reasons.

## 8. Report

`reports/phase16/agent_04_report.md`, sections mirroring this file. Keep the
Phase 15 honesty posture: the belief-vs-count history is a run of null results;
if this one is null too, say so in the first paragraph.
