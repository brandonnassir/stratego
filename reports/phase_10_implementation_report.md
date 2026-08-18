# Phase 10 Implementation Report

Phase 10 is a learned **setup-selection** phase. It asks whether game
outcomes can be used to learn a better distribution over the frozen
Phase 7 setup library while preserving setup diversity, information
safety, reproducibility, and the accepted Phase 9 move model. The move
policy is not retrained: `checkpoints/phase9/selfplay_c1_v1.pt` must be
byte-identical before and after the phase.

## 1. Agent 1 — Contract, Seeds, Banks, and Acceptance Freeze

**Status: PASS** — 22/22 completion
gates true, zero problems, zero Phase 10 outcome games, zero utility fits,
zero C1 optimizer steps.

Agent 1 freezes the entire Phase 10 experiment before any outcome exists.
Nothing below was chosen after seeing a result, because no Phase 10 result
exists yet.

### 1.1 Verified upstream identities

Every identity was recomputed from live bytes, not read from a record.

```text
Phase 9 Agents 1-8              all PASS, zero false completion gates
Phase 9 checkpoint SHA-256      dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
Phase 9 model-state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 9 parameters              863,959, all finite
C1 config digest                31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d
Phase 9 contract digest         ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34
Phase 9 amendment v1 digest     ee4b05078c676128f78c8e5c31bd10ce4f0841e34a57c4c7c3fca6616e083ac4
Phase 9 amendment v2 digest     92ad4f67fb07a14551ef555335b71000d6369cd817dad59c839d793888de9e71
Phase 7 library content         7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
Phase 7 library metadata        d86f486182a820d546d470ef4ebce92ff60c6259aed80c481bc985bce8c64980
Phase 7 library manifest        53139ab7e21c4e8a31987507d6fb1eabf93f36cdc1221fe85d08042963488f31
splits                          6,400 / 800 / 800 at 400 / 50 / 50 per family
trait vectors                   8,000 / 8,000 reconstruct exactly
neutral_v1                      reflection 0.5, perturbation 0.5, uniform 1..6
pre-existing Phase 10 work      none: no corpus, utility, candidate or selector
```

### 1.2 Frozen contracts

Eight documents, canonical JSON, SHA-256.

```text
phase10_setup_contract_v1         94a1d17161fc936b8f11ed10289fe3fd4aed7bab484dac927d5baa035cc935ad
phase10_setup_outcome_corpus_v1   951025f102dab1a103d02f21e5df414265bc594b37bd02283a64fe02585fe6d5
phase10_setup_utility_v1          2778ddea8bb1c85b998a3abaefaf794816bc9b6eb476010b44d040087758f456
phase10_setup_selector_v1         8a3459fbfb88a45f207fe0965dd6c743524ef16168a78b8ab748ff4efd2bd0b2
phase10_selector_schedule_v1      30ad8ede3fe342d071a5a5d7dc65510bf6cdea3ff20c70554d3e181d97b86dc4
phase10_eval_bank_v1              8e4158426e783f55590086164e9e5fccbd331373b04e1e36a9b7358aaf87f22b
phase10_acceptance_v1             a76f79b7a710f327d2ee097aa922203f1e19ec7bb7619d5baac559e73af7e88b
phase10_system_v1                 a8b44e1a12bcc31ed446d031c188129dc82584ed64086601ed9b9edb7830a48d
bundle                            1cfa5b4667bb75bfb9b323f450ec23d5f812dba629e80a9bce0b19dabb02b395
```

`phase10_system_v1` binds what exists now — the accepted Phase 9 move
model, the frozen Phase 7 reflection/perturbation path and `neutral_v1` —
and leaves three slots unbound with their filling rules: the accepted
utility model, the accepted trait scaler and the selected selector config.
Inventing values for those now would be exactly the pre-commitment the
phase forbids; Agent 6 fills them at the production freeze.

### 1.3 Seeds and derivations

```text
master                    2026081801     outcome-corpus schedule   2026081802
setup draws               2026081803     utility fitting           2026081804
selector/candidate draws  2026081805     validation/case schedule  2026081806
validation bootstrap      2026081807     final-test bootstrap      2026081808
```

All nine streams derive through `blake2b(person='strat-s10')` over
`identity_version:domain:domain_root:parts`, a tag disjoint from every
accepted upstream tag. No derivation reads worker count, arrival order,
process id, wall clock or a storage path.

The collision audit enumerated 58,792 seeds across the frozen
id space and found 58,792 distinct values — zero duplicates inside a
stream and zero collisions across streams.

### 1.4 The 16,384-game outcome schedule

```text
256 ordered family pairs x 64 games = 16,384
schedule digest   1a49f05032e300a8ecef81aa09776ed0d0766149576afb8eaa74a97e974e98b0
split             train only; zero held-out bases
side draw         first attempt whose neutral_v1 draw matches the scheduled family
move behaviour    accepted Phase 9 checkpoint both sides, greedy float32,
                  single_request, no search, zero optimizer steps
```

Ordering is a real distinction: `(F03, F11)` and `(F11, F03)` are two of the
256 scheduled pairs, which is what lets the fit separate the red-first
intercept from setup quality. Counts are arithmetic, never sampled, so the
corpus shape cannot depend on a seed or a worker count.

### 1.5 Utility definition

```text
Model F   u_F(s, c) = b_eff[c, family(s)]                    33 parameters
Model T   u_T(s, c) = b_eff[c, family(s)] + w[c] . x(s)     127 parameters
features  phase10_trait_feature_v1, 47 float64 scalars
scaler    phase10_trait_scaler_v1, train-only, ddof=0
          fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9
objective mean BCE over the 16,384 games + 1e-3 * sum of squares on the raw
          family offsets and trait weights; the intercept is unpenalized
optimizer float64 CPU L-BFGS, lr 1.0, max_iter 500, history 50,
          tol_grad 1e-10, tol_change 1e-12, strong_wolfe, all-zero start
```

`strong_wolfe` line search is available in this environment (torch 2.13),
verified from live bytes before the freeze, so no deterministic-equivalent
authorization is needed. The utility domain is the *base*, never the played
arrangement: a selector chooses a base and only then hands it to the frozen
reflection/perturbation path, so fitting on base identity is the only choice
that keeps the six legal selector inputs legal.

### 1.6 Exactly six candidates

```text
P10-A model_F T=0.75    P10-B model_F T=1.25    P10-C model_F T=2.00
P10-D model_T T=0.75    P10-E model_T T=1.25    P10-F model_T T=2.00
```

All six share the frozen 0.35 neutral / 0.65 learned mixture. The two
utility models are fit once; candidate-specific refitting is forbidden;
`neutral_v1` is the baseline and never a seventh candidate.

### 1.7 The two evaluation banks

A Phase 9 case fixed both setups because both sides were policies. A
Phase 10 case cannot: the experiment is about which setup a selector
chooses. A case therefore fixes one held-out opponent setup, two selector
draw seeds (one per colour, identical for the learned candidate and the
neutral baseline), the two `neutral_v1` own-side draws those seeds produce,
and per-matchup match seeds that are independent of arm and candidate. The
selector under test plays Red in game 0 and Blue in game 1 against the same
opponent setup; the bootstrap unit is the case.

```text
phase10_validation_bank_v1   128 cases, validation split, 8/family
  bank digest      a37ff113d03a0f67e760e447a462cc0d0d8de83f063d395715aeb77be355657f
  manifest digest  459cef36d7032beb8fc9665efa7692dac3c40c68109e9f0bcdefa6141bd0906e
phase10_test_bank_v1         512 cases, test split, 32/family
  bank digest      be04b891ba5ab142aacbd937ab24f79054843310e8d28f6b8cbee65daef931ad
  manifest digest  c6f21bcdb829fe77b208e49d9960b05a1b65bcf1dc7944d3f10420bea132a755
```

### 1.8 Isolation

Phase 10 does not claim a wholly unseen base-template universe — earlier
phases already used the same held-out base pool — so it claims what it can
prove.

```text
Phase 9 held-out fingerprint set   1,184 arrangements
  set digest                       c714c6e721e65d2624b34b27a529fa95f69369d0f1070d31b134d1b69aac16ce
frozen Phase 10 arrangements       1,920 (opponent + both neutral own-side draws per case)
overlap with Phase 9                0
validation-test overlap             0
within-case duplicate fingerprints  0
```

The rejection walk is not decorative: it fired on 7 of 256 validation
selector seeds and 141 of 1024 test selector seeds, which is the unperturbed
branch colliding with Phase 9's held-out draws exactly as expected. Both
walks read only quantities fixed before any selector exists, so they are
arm-independent and order-independent, and a case rebuilds alone.

One residual is recorded rather than hidden: a learned selector's own-side
draw cannot be enumerated before the selector exists. Rejecting such a draw
at evaluation time would distort the very mixed distribution the diversity
contract is stated over, so Agents 5-7 record its Phase 9 landings as a
report-only diagnostic instead.

### 1.9 Acceptance

All eight gates are hard. Strict and non-strict thresholds are named
separately in code (`above` vs `at_least`), and each was exercised at its
boundary and one representable step on the failing side.

```text
A  direct        EWR >= 0.49, LB > 0.47      improved: EWR >= 0.52, LB > 0.50
B  league        Delta_L >= -0.01, LB > -0.03    weights .45/.35/.20
C  individual    per-opponent paired LB > -0.03
D  easy          Random >= .95 / Red >= .90 / Blue >= .90, Basic >= .80,
                 paired LB > -0.03
E  diversity     every threshold over the final mixed distribution
F  correctness   nine counters, all exactly zero (a missing counter fails)
G  reproducible  id + seed + identity + split + colour -> same fingerprint
H  preservation  exact Phase 9 SHA, state digest, parameters, zero steps
```

Statistics: paired-unit percentile bootstrap over NumPy PCG64, 10,000
replicates, 95%, one domain-separated stream per matchup and per
difference, resampling the logical case so a case's two colour games move
together.

### 1.10 Recorded readings

- **case-schedule seed root** — the contract says *validation cases: 2026081806*.
  The contract names a validation-case root but no separate test-case root,
  so 2026081806 roots the case schedule of both banks, domain-separated by
  their two distinct bank versions. No root is added, no stream is reused,
  and the exhaustive collision audit proves every validation and test stream
  disjoint.
- **trait feature dimensionality** — the contract says *x(s) is the frozen 35-field trait vector*.
  The 35 frozen fields include four per-rank histograms, so the feature
  vector is their lossless flattening: 47 float64 scalars, nothing dropped
  and nothing invented. The alternative would require discarding schema
  information by hand; the 16 exact linear relations this surfaces leave
  rank 31, and the frozen L2 penalty of 1e-3 makes the minimizer unique.
- **objective reduction** — the contract says *full-batch BCE + L2 1e-3 on family/trait parameters*.
  BCE is the mean over the 16,384 scheduled games and the penalty is lambda
  times the sum of squares of the raw family offsets and trait weights; the
  intercept is unpenalized. A summed BCE would make 1e-3 effectively no
  regularization at 16,384 games; the mean is the reading under which the
  stated coefficient does the job the contract gives it.
- **fingerprint isolation of learned draws** — the contract says *zero exact final-setup fingerprint overlap with Phase 9 validation/test cases*.
  Hard, rejection-enforced over every arrangement a Phase 10 case fixes —
  the opponent setup and both neutral_v1 own-side draws; a learned
  selector's own-side draw cannot exist before the selector does, so Agents
  5-7 record its Phase 9 landings as a report-only diagnostic. Rejecting a
  learned draw at evaluation time would distort the very mixed distribution
  the diversity contract is stated over, and the diagnostic keeps the
  residual visible rather than unmeasured.

### 1.11 Evidence

```text
tests before   4621 passed, 3 skipped in 283.69s (0:04:43)
tests after    4858 passed, 3 skipped in 295.45s (0:04:55)
```

```text
reports/phase_10_data/agent_01_setup_selection_contract.json
reports/phase_10_data/agent_01_validation_bank.json
reports/phase_10_data/agent_01_test_bank.json
reports/phase_10_data/agent_01_acceptance.json
```

### 1.12 Handoff to Agent 2

Agent 2 collects `phase10_setup_outcome_corpus_v1` and makes no
learning-design decision. It receives the schedule enumerator and
rebuilder, every contract and schedule digest, the setup derivations, the
outcome-record schema, the Phase 9 evaluation-only identity, the
resolver/storage policy, the exact 16,384 logical ids, the train-only rule,
the crash/resume identity rule, and the Phase 9 byte-preservation
requirement.

