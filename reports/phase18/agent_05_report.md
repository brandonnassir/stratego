# Phase 18 Agent 5 — G2 publication, the P18-D004 correction, and the bounded raw-actor confirmation

**Status: complete. Decision `P18-D005` = `PROCEED`** for the synthetic
trainability portion of Gate G2 only. The reviewed G2 branch is published at
`6afa13be`; the P18-D004 wording is corrected with every number intact; and on
an independently generated landscape with fresh seeds and the G2 learning
method unchanged byte for byte, the raw generation actor learned in every seed
(pooled paired 95% [+5.2985, +5.5585] on 12,288 pairs) and the median seed
closed 10.007% of its initial-to-optimum gap against the frozen 10% threshold.
Not pushed — awaiting review.

## Part 1 — publication of the approved Agent 4 work

- `phase18/g2-setup-parity` resolved to exactly
  `6afa13bed355884a3327d2661fd739784260dc2b`; the only pre-existing
  uncommitted change was the protected
  `reports/phase13/phase14_launch_manifest_v1.json`, which was never staged,
  edited, restored, stashed or deleted (every commit used explicit paths).
- Published with a normal non-force push at 2026-09-02T14:28:00Z; the remote
  branch was absent beforehand; local and remote both resolve to `6afa13be…`;
  no publication commit was added.
- `phase18/g2-raw-confirmation` was created from that exact commit.

## Part 2 — the G2 documentation correction

`P18-D004 = REVISE` is preserved. In `P18-D004.md`, `P18-D004.json` and
`agent_04_report.md` the claim that the EMA criterion was unreachable by
arithmetic regardless of learner behaviour is withdrawn; the packet now states
that decay 0.999 retained `0.999^64 = 0.937975` of the initial parameter
contribution (an approximately 1,000-update time constant), that the frozen
64-update assay empirically showed severe EMA lag (the raw actor learned
strongly while the EMA captured 1.3–2.8% of the observed change), and that the
development smokes informed the raw diagnostic and the instrument-defect
interpretation without changing the frozen landscape, budget, threshold or run
seeds. No result, identity or number changed; the original wording of every
corrected sentence is preserved in the JSON packet's `corrections.items`; the
frozen G2 contract and result artifacts are untouched. The review record
`reports/phase18/reviews/P18-D004_REVIEW.md` carries the acceptance, the
publication receipt, the corrections with as-reviewed and corrected digests,
and the authorized next question. The operator's work package is saved
verbatim as instruction 07 and bound in both indexes. Committed at `59ddc75`.

## Part 3 — what was frozen (before any outcome)

`scripts/phase18_g2_raw_confirmation.py` and its nine tests, the contract
`phase18_g2_raw_confirmation_contract_v1.json` and the landscape document were
committed at `G2_RAW_SOURCE_COMMIT ccddceda27015f47d26879802b4b55653c8fdf18`
(tree `dd9d305e…`):

```text
seed namespace        phase18_g2_raw_confirmation_v1:6afa13bed355884a3327d2661fd739784260dc2b
seed rule             derive_stream_seed(seed_namespace, label, *parts), one label per stream
model seeds           3224018716491833412 / 3702536526967685428 / 6940975017500961852
landscape table seed  475578596360375388   table digest 210352ef…
exact optimum         53.031081  (LP duality; potentials recorded; re-verified by arithmetic)
uniform baseline      mean -0.342275, SD 5.952866 (Hoeffding)
outcome mapping       P(win) = 0.9 sigmoid(3z), P(draw) = 0.10, 4 outcomes per eligible setup
bootstrap seeds       raw pooled 2538560817612476064; raw per-seed, EMA pooled and EMA
                      per-seed draws separately labelled
freshness audit       no table, model, pool, evaluation or bootstrap seed shared with the
                      G2 assay or either development smoke (all rebuilt and compared)
method identity       16/16 learning-method files byte-identical to the G2 launch manifest;
                      design identical on every method field; configuration identical
                      once the run id is removed
primary endpoint      the RAW generation actor at 0 and 64 updates (this assay only)
EMA                   secondary telemetry at the same endpoints; cannot change the decision
decision rule         PROCEED iff parity, replay, binding and integrity pass, raw final >
                      initial in all three seeds, pooled paired 95% lower bound > 0, median
                      raw gap closure >= 10%; STOP if valid and failing; REVISE only for a
                      named implementation or measurement defect
```

## Part 4 — what ran, in order

```text
59ddc75   documentation correction, instruction 07, review record, indexes, .gitignore
ccddceda  G2_RAW_SOURCE_COMMIT: driver, tests, contract, landscape (freeze)
e980dc7   preflight in the canonical checkout at ccddceda: method digests identical to the
          G2 launch manifest; 86/86 evaluator; 131/131 setup (9 new); oracle PASS on this
          run's seed; coverage 30/30
77ce90b   launch manifest from the clean detached worktree gpt_agent_phase18_g2_raw_exec
          at ccddceda (porcelain empty, base commit proved ancestor, zero outcomes)
          seeds 1, 2, 3 run concurrently from the worktree, CPU float32, 4 threads each,
          913 / 918 / 912 s; artifacts under artifacts/phase18/g2_raw_confirmation_v1
          (git-ignored, canonical tree; 99 MB)
          --replay (bitwise on all endpoints), --analyse, --bind (11 artifacts, one
          source commit), --decide (frozen rule -> PROCEED)
be4fc84   result evidence: per-seed results, replay, results, binding ledger, decision input
(next)    this report, P18-D005, index and documentation updates
```

Every seed completed 64 updates, 320 optimizer steps and 64 EMA updates.
Zero legality, orientation, attribution, non-finite or checkpoint-identity
events; all twelve held-out utility arrays hold exactly 4,096 finite values;
zero immediately terminal setups in any pool or evaluation sample (the S24
exclusion never fired); zero duplicates collapsed.

## Part 5 — results

| seed | raw initial | raw final | raw paired 95% | raw gap closed | EMA initial | EMA final | EMA gap closed |
|---|---|---|---|---|---|---|---|
| 1 | −1.1436 | +4.6249 | [+5.545, +5.988] | **10.648%** | −1.1436 | −1.0902 | 0.099% |
| 2 | −0.5994 | +4.7675 | [+5.140, +5.601] | **10.007%** | −0.5994 | −0.5356 | 0.119% |
| 3 | −0.6146 | +4.5347 | [+4.926, +5.375] | **9.599%** | −0.6146 | −0.5100 | 0.195% |

```text
raw pooled paired delta   +5.4282   95% [+5.2985, +5.5585]   n 12,288   median gap 10.007%
EMA pooled paired delta   +0.0740   95% [+0.0116, +0.1386]   n 12,288   median gap  0.119%
EMA / raw displacement    0.9% / 1.2% / 2.0%      0.999^64 = 0.937975 retained
raw z initial -> final    -0.135 -> +0.834   -0.043 -> +0.858   -0.046 -> +0.819
```

The frozen raw criteria all pass; the median gap closure clears the 10%
threshold by 0.0073 percentage points of the gap (about 0.004 utility units on
the median seed), with seed 3 individually below it. The learning signal itself
is unambiguous (pooled lower bound 0.89 uniform SDs above zero). The EMA
telemetry improved in every seed, its pooled lower bound is barely above zero
and every per-seed interval includes zero; it decided nothing.

Compared with the G2 raw diagnostic (20.9%, 18.5%, 14.8%; pooled delta
+10.77), the direction and reliability replicate while the magnitude is about
half (+5.43): this table starts the fresh policy higher (z ≈ −0.05 to −0.13
versus −0.19 to −0.76) and has a lower optimum (53.03 versus 55.62), and the
raw actor reached z ≈ +0.83 rather than ≈ +1.3 in the same 64 updates. The
gap-closure fraction is therefore landscape-dependent, which is why it sits at
the threshold's edge here. Learning-curve telemetry matches G2: the raw curves
rise monotonically apart from one dip (seed 3, update 48) and are still rising
at update 64; the EMA curves are flat within noise until the last points.

## Part 6 — what did not run

- No Stratego game, no setup-learning game, no Phase 8 warmstart training, no
  G3 work, no tandem pilot, no sealed Phase 8 access, no tuning of any kind.
- No frozen field, source file, threshold, budget, seed or landscape changed
  after the first outcome; no namespace was abandoned; no seed was rerun.
- No push of `phase18/g2-raw-confirmation`.

## Part 7 — reading

- **Observed.** Method identity to G2 by digest; parity on every row and in
  the oracle; strong, consistent raw-actor learning on all three seeds; the
  median gap closure at 10.007%; EMA lag of 1–2% of the raw displacement;
  bitwise replay; one bound source commit.
- **Supported.** The parity-correct raw learner reliably learns an
  independently generated landscape within 64 updates. The G2 raw-actor
  finding replicates. `PROCEED` under the frozen rule, for the synthetic
  trainability portion of G2 only.
- **Plausible, untested.** The gap-closure fraction depends on the table;
  another draw could land either side of 10% under the same learning
  strength. A longer budget would close more and eventually carry the EMA.
- **Not supported.** That the method learns Stratego setups; that the EMA at
  this budget can serve as an evaluation model; that the 10% threshold is
  cleared robustly; any authorization beyond designing the next gate.

## Recommended next question

Design, not launch: what evaluation model and update budget give Gate G3 an
instrument with power? The EMA needs a budget on the scale of its ~1,000-update
time constant (or a predeclared, reviewed alternative for a pilot); the next
margin should be sized in the units the gate is scored in from a known
instrument resolution rather than as a fraction of a landscape-dependent gap;
and the evaluation contract's training-versus-evaluation rule wording
(O-P18-EVALRULES-1) must be amended before any real-game measurement. The
first Stratego-facing question would then be whether the setup learner,
trained on real outcomes against the frozen Phase 8 teacher schedule, improves
over its fresh initialization on a paired setup-development pack.

## Stop state

Stopped after committing on `phase18/g2-raw-confirmation`. Worktree
`gpt_agent_phase18_g2_raw_exec` at `ccddceda` remains registered. The
unreviewed result stays local until `P18-D005` is reviewed.
