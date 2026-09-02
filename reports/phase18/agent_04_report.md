# Phase 18 Agent 4 — G1 publication, Gate G2 setup parity and the synthetic assay

**Status: complete. Decision `P18-D004` = `REVISE`.** Every parity requirement
passes (30/30 method-map rows, the independent oracle, 208/208 tests, zero
integrity events), the learner demonstrably learns the frozen synthetic landscape
on all three seeds (raw actor: 20.9%, 18.5%, 14.8% of the initial-to-optimum
gap closed), but the gate's frozen evaluation model — the EMA at decay 0.999
updated once per setup update — retains 93.8% of its initial parameters after
the 64-update budget and closes 0.28%, 0.52%, 0.35% of the gap. The EMA
criterion cannot be met at this budget whatever the learner does. That was
predeclared in the contract before any frozen seed ran, with the interpretation
that decides this packet. Not pushed — awaiting review.

## Part A — publication and the source boundary

- `origin` fetched without merging. The local `phase18/g1-random-confirmation`
  resolved to the approved `ef7523c1940650c0906d1927e64679e8328a663f`; the
  remote branch was absent; the P18-D003 JSON/Markdown hashed to the approved
  `57281f6a…` / `7d930357…`.
- Published with a normal non-force push. Local and remote both resolve to
  `ef7523c1940650c0906d1927e64679e8328a663f`; no publication commit was added.
- `phase18/g2-setup-parity` created from that exact commit (it existed neither
  locally nor remotely).
- The protected `reports/phase13/phase14_launch_manifest_v1.json` modification
  was never staged, edited, restored, stashed or deleted; every commit used
  explicit paths.

## Part B — review and erratum

`reports/phase18/reviews/P18-D003_REVIEW.md` records acceptance, G1 closure,
the independent reproduction of the 4,096-pair result from the receipts, and one
erratum: `P18-D003.json` `identity.rules` says `battleless_move_limit 100`, but
the engine's `EVALUATION_RULES` constant, the driver, the frozen contract's rules
token and all 16,384 receipts carry the accepted evaluation value **200**.
`reports/phase18/phase18_rule_identity_errata_v1.json` records it as a narrative
metadata error (no rerun, packet not rewritten) and records the separate open
item that `phase18_evaluation_contract_v1.json` names *training* rules (100)
for the future play lanes — preserved unedited, **amendment required before any
real-game G3/G4 evaluation**. The instruction was saved as
`06_AGENT_4_G2_SETUP_PARITY_AND_SYNTHETIC_ASSAY.md` (SHA-256 `dd390ba5…`); the
decision and instruction indexes, the Phase 18 manifest, STATUS and the evidence
index were updated and committed at `7460e3d` before any implementation.

## What was built (Parts C–F)

All new code lives in `stratego/training/phase18/` (contract, model, sampling,
buffer, learning, reference oracle, synthetic landscape, synthetic assay,
coverage), `tests/training/phase18/` (122 tests) and
`scripts/phase18_g2_setup_parity.py`. Phase 17 code is neither edited nor
imported on the training path; the accepted orientation helper, setup identity
frame, engine setup validators and parameter digest are imported unmodified.

Every method-map row S01–S30 is implemented and mapped to symbols and tests in
`coverage.py`; `phase18_g2_parity_coverage_v1.json` marks a row complete only
from the recorded JUnit outcome of every cited test (30/30). The mandatory
semantics are mechanisms, not notes:

```text
entropy target         I/10                         setup_buffer.process, S12
advantage residual     I - 10h                      setup_buffer.process, S13
outcome aggregation    running mean of one-hot      setup_buffer.add_outcome, S09
zero outcomes          not ready, never a draw      S09 / S24
identity               played-board fingerprint,    setup_buffer.add_pool, S10
                       newest snapshot wins
window                 counts reset at each pool    S23
retention              storage window, fatal lookup S21 / S10
handedness             Flag on files 5-9 only       setup_sampling.legal_masks, S04
reflection             independent seeded 50%       setup_sampling.generate_pool, S05
orientation            flip back before any gather  setup_buffer._batch, S06
engine boundary        accepted helper only         setup_sampling.to_engine_setup, S07
PPO / KL / weights     0.2 / KL(cur||beh) 0.1 /     setup_learning.setup_batch_loss,
                       0.5 value / 1.0 entropy      S16-S18
lambda                 pinned 1.0, refused else     S15
optimizer              AdamW lr 5e-5 wd 0           SetupTrainer, S25
batch / epochs         1,024 per step / 5           SetupTrainer.update, S26
clipping               0.5 on setup params          S27
EMA                    0.999 once per update;       SetupEMA, S28
                       raw generates, EMA evaluates
checkpoint             raw / optimizer / EMA files  S29 (round trip on CPU and MPS)
```

`reference_oracle.py` recomputes every quantity of the published update in
numpy float64, including the authors' TD/GAE recursion, and never imports the
production loss or buffer (a test scans its imports). The recorded oracle run
(`phase18_g2_parity_oracle_v1.json`): every loss term agrees with the
production loss on a float64 model copy to 1.8e-15; thirteen representative
gradients (heads, embeddings, attention, feed-forward, layer norms) agree with
central finite differences to 2e-10; AdamW at zero decay reproduces the oracle
Adam update exactly; the clip scale is exact and the post-clip norm is 0.5; the
EMA follows its closed form to float32 precision; step counts match the
published loop; raw, optimizer and EMA restore on CPU and on MPS and take one
further update.

## The frozen assay (Part G)

`phase18_g2_contract_v1.json` and `phase18_g2_synthetic_landscape_v1.json` were
committed at the first freeze (`861fac8`) and are byte-identical in
`G2_SOURCE_COMMIT`. The landscape is a reflection-symmetric additive
piece-by-square table (seeded from the namespace), with an exact optimum
**55.6234** certified by LP duality and exact uniform-random moments (mean
−0.8136, SD 6.1561, Hoeffding). Outcomes: `P(win) = 0.9·sigmoid(3z)`,
`P(draw) = 0.10`, one seeded uniform per (period, setup, replicate); four
outcomes per eligible setup; the learner receives outcomes only. Three model
seeds derive from `phase18_g2_setup_parity_v1` through `derive_stream_seed`;
64 updates; 1,024-setup pools; 4,096 held-out samples per endpoint with common
random numbers; 10,000-replicate paired bootstrap; 10% median gap closure;
final-update checkpoint; exact decision interpretations.

**Predeclared instrument finding.** Two development smokes on separate
namespaces (`reports/phase18/g2/dev_smoke_v1.json`) and the decay arithmetic
established, before any frozen seed ran, that an EMA at 0.999 updated 64 times
retains `0.999^64 = 0.9380` of its initial parameters and moves only about 3%
of the raw actor's displacement, so the EMA-measured criteria have almost no
power at this budget. The contract therefore records the raw actor's held-out
utility on the same evaluation stream as a **diagnostic that decides nothing**,
and predeclares: EMA criteria pass → PROCEED; EMA fails while the raw
diagnostic satisfies the same three criteria → REVISE (instrument defect
"EMA horizon exceeds the update budget"); both fail → STOP. No frozen field was
chosen from the smokes.

## Execution (Part H)

```text
G2_SOURCE_COMMIT   354a4cad55a88dca6dcb24a21cf79cecc130008f  (tree 0bbccb82b21f5988…)
verification       11b5558  (86/86 evaluator, 122/122 setup, oracle PASS, 30/30 rows)
launch manifest    6f5297e  (written from the clean worktree, porcelain empty)
worktree           /Users/brandonwashington/Dev/Github/stratego/gpt_agent_phase18_g2_exec
artifact root      artifacts/phase18/g2_setup_parity_v1  (git-ignored, canonical tree)
seeds              1, 2, 3 run concurrently, CPU float32, 4 threads each, ~925 s per seed
```

Every seed completed 64 updates, 320 optimizer steps, 64 EMA updates. Per
update: 1,021–1,024 ready rows (12 immediately terminal setups excluded across
the run, never as draws), exactly four outcomes per ready row, 1,024 distinct
fingerprints per pool, reflected fraction 0.499–0.502, Flag in the permitted
half 100% before reflection, zero duplicates. **Zero** legality, orientation,
attribution, non-finite or checkpoint-identity events. No Stratego game was
played and no sealed Phase 8 example opened.

Deterministic replay (`phase18_g2_replay_v1.json`, a separate script written
after the launch): the landscape rebuilds to its digest, optimum and moments;
the design digest equals the launch manifest's; for every seed the initial raw
model rebuilds to its digest, the first period's 1,024-setup pool and all 4,096
outcomes replay exactly from the seeds, all 64 period digests re-derive from the
receipts, and both evaluation endpoints regenerate **bitwise** (max abs
difference 0.0) from the initial seed and from the three-object checkpoint.

## Results (Part I)

Held-out expected utility (mean landscape utility of 4,096 EMA samples):

| seed | EMA initial | EMA final | gap closed | paired 95% (EMA) | raw initial | raw final | gap closed (raw) |
|---|---|---|---|---|---|---|---|
| 1 | −5.5082 | −5.3385 | +0.28% | [+0.079, +0.263] | −5.5082 | +7.2805 | +20.92% |
| 2 | −3.7918 | −3.4842 | +0.52% | [+0.176, +0.440] | −3.7918 | +7.2010 | +18.50% |
| 3 | −1.9667 | −1.7658 | +0.35% | [+0.080, +0.322] | −1.9667 | +6.5673 | +14.82% |

```text
EMA  pooled paired delta   +0.2261   95% [+0.1588, +0.2949]   n 12,288   median gap  0.35%
raw  pooled paired delta  +10.7718   95% [+10.6338, +10.9089]  n 12,288   median gap 18.50%
EMA retained initial fraction   0.999^64 = 0.9380
```

The frozen EMA criteria: all three seeds improved (pass), pooled lower bound
strictly above zero (pass), median gap closure ≥ 10% (**fail**, 0.35%). The
raw diagnostic satisfies all three. The EMA's movement is 1.3%, 2.8% and 2.4% of
the raw displacement per seed — the predicted few percent.

Learning-curve telemetry (all seeds alike): the entropy head converges to
`I/10` by update ~16 (entropy-prediction loss 11–12 → 0.3 → 0.1), the
entropy-to-outcome absolute-magnitude ratio falls from 3.8–4.1 to 0.17–0.20,
the pool's mean utility z-score rises monotonically (−0.76 → +1.34,
−0.53 → +1.33, −0.18 → +1.19) and is still rising at update 64, the value
head tracks the outcome mean (E[v] ≈ z̄ at the end), mean prefix entropy stays
at 1.6 nats (no collapse), the final-epoch reverse KL never exceeds 0.0012, the
PPO clip fraction is 0 throughout, and the gradient clip is active on every
step (pre-clip norm 34–38 at the start, 2–3 at the end).

## Reading

- **Observed.** Parity holds on every row and in the oracle; the learner
  improves the landscape utility on all three seeds with tight, non-overlapping
  paired intervals on the raw actor; the EMA moves by a few percent of that.
- **Supported.** The scaled setup-policy implementation matches the published
  method at loss, gradient, sampling, aggregation, optimizer, checkpoint and EMA
  semantics, and learns a known landscape from outcome-only feedback. The gate's
  frozen evaluation model cannot reflect that learning inside 64 updates, by
  arithmetic and as observed.
- **Plausible, untested.** The raw curve is unconverged at 64 updates; a longer
  budget would eventually carry the EMA. The entropy bonus dominates the first
  ~8 updates and outcome-driven learning accelerates once `h` converges.
- **Not supported.** That the method learns *Stratego* setups; that the raw
  diagnostic satisfies the frozen gate (it was never the decision model); any
  claim about a different budget or hyperparameter.

## Deviations

1. Raw-actor and pool-utility diagnostics were added to the runner after the
   first smoke and before the freeze; they decide nothing and the contract says
   so. Two development smokes ran on other namespaces and are recorded.
2. Two driver-only corrections landed between the first freeze commit
   (`861fac8`) and `G2_SOURCE_COMMIT`: the oracle's EMA closed-form tolerance
   (1e-7 → 1e-5, float32 shadow) and an indexing slip in the oracle's
   `I − 10h` cross-check (batch row versus processed row). The frozen contract
   and landscape files did not change.
3. The `--verify` stage ran in the canonical checkout (dirty only by the
   protected manifest and its own outputs) because the evaluator suite reads
   git-ignored accepted checkpoints that live there; the launch manifest and
   all three seeds ran from the clean worktree.
4. The three seeds ran concurrently in separate processes; determinism is
   per-process (pinned threads) and the replay reproduces both endpoints bitwise.
5. The replay script and this report are new files not covered by the launch
   manifest's digests; they change nothing the assay executed through.

## Commit trail (all local on `phase18/g2-setup-parity`, none pushed)

```text
7460e3d  review, erratum, instruction 06, indexes (Part B)
861fac8  first freeze: implementation, tests, contract, landscape, dev smokes
81fab3a  oracle tolerance / verification record
354a4ca  G2_SOURCE_COMMIT: oracle cross-check indexing (final pre-run source)
11b5558  verification artifacts (JUnit x2, coverage, oracle, record)
6f5297e  launch manifest bound to 354a4ca from the clean worktree
6623621  results, per-seed summaries, replay, binding ledger
(next)   this report, P18-D004, index and documentation updates
```

## Stop state

Stopped after committing. No G3 work, no Stratego setup-learning game, no push.
The unreviewed G2 result stays local until `P18-D004` is reviewed.
