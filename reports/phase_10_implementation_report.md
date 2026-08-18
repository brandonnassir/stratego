# Phase 10 Implementation Report

Phase 10 is a learned **setup-selection** phase. It asks whether game
outcomes can be used to learn a better distribution over the frozen
Phase 7 setup library while preserving setup diversity, information
safety, reproducibility, and the accepted Phase 9 move model. The move
policy is not retrained: `checkpoints/phase9/selfplay_c1_v1.pt` must be
byte-identical before and after the phase.

## 1. Agent 1 — Contract, Seeds, Banks, and Acceptance Freeze

**Status: PASS** — 23/23 completion
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
phase10_setup_contract_v1         058df41f304a2b2225222bf69df7b462bb90426ca0924992f792f5c7bcb1c71a
phase10_setup_outcome_corpus_v1   951025f102dab1a103d02f21e5df414265bc594b37bd02283a64fe02585fe6d5
phase10_setup_utility_v1          2778ddea8bb1c85b998a3abaefaf794816bc9b6eb476010b44d040087758f456
phase10_setup_selector_v1         5e2b9c3a0192215545ba5c0d7164e4833d7c77dd27a3209f7d81bab6037f3efe
phase10_selector_schedule_v1      30ad8ede3fe342d071a5a5d7dc65510bf6cdea3ff20c70554d3e181d97b86dc4
phase10_eval_bank_v1              8e4158426e783f55590086164e9e5fccbd331373b04e1e36a9b7358aaf87f22b
phase10_acceptance_v1             a76f79b7a710f327d2ee097aa922203f1e19ec7bb7619d5baac559e73af7e88b
phase10_system_v1                 a8b44e1a12bcc31ed446d031c188129dc82584ed64086601ed9b9edb7830a48d
bundle                            257f140dadddc00e4f75217ecedfe726390167de8769db0b5c40021e4388612f
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

Those are the **eight root seeds**. Beneath them sit **ten derived
domains** — a distinct one per randomness need, several sharing a root —
so "streams" in this report always means derived domains, never seeds:

```text
corpus_setup     corpus_match     bank_opponent    bank_selector
bank_match       selector_branch  selector_base    selector_audit
utility_fit      bootstrap
```

All ten derive through `blake2b(person='strat-s10')` over
`identity_version:domain:domain_root:parts`, a tag disjoint from every
accepted upstream tag, so two domains sharing a root still cannot
collide. No derivation reads worker count, arrival order, process id,
wall clock or a storage path.

`selector_audit` was added during Agent 1 review reconciliation, on the
one randomness need the first freeze left unfrozen: Agent 4 must run
100,000 selector draws per candidate x colour x split, addressable by
draw id, with resume as exact set subtraction by draw id — and nothing
produced that draw's selector seed. `case_selector_seed` covers only the
1,280 bank-case seeds. Leaving it open would have made Agent 4 invent a
derivation Agent 1 owes it. No root seed was added or changed; the domain
hangs off the existing `selector_draw_seed` 2026081805, and consecutive
audit ordinals receive unrelated hashed streams rather than adjacent
integers.

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

The accepted Phase 9 held-out universe can be counted two ways, and both
counts are correct. A Phase 9 pair stores each side already oriented for
the player that plays it, and the Red and Blue orientation maps differ, so
one canonical arrangement appearing on a Red side in one case and a Blue
side in another is *two* stored board strings and *one* canonical
identity. That is the whole difference:

```text
held-out sides                     1,280
distinct stored engine boards      1,233
distinct canonical identities      1,184   (49 of them seen in both orientations)
  1,184 + 49 = 1,233
```

The isolation set is stated over canonical final-setup fingerprints
because that is the accepted Phase 7 setup identity and the thing a
Phase 10 case actually produces. `phase9_raw_board_coverage` is the
receipt that the canonical statement loses nothing: every stored board is
de-oriented by the player that played it, run through the exact Phase 10
fingerprint function, and required to land in the set.

```text
raw boards mapped                  1,233 / 1,233
unmapped raw boards                0
round-trip mismatches              0
identities never reached           0   (the map is onto the whole set)
duplicate classes                  49, every one of size exactly 2
```

```text
isolation set                      1,184 canonical identities
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
contract is stated over, so it stays a report-only diagnostic. Agents 5-7
carry the standing obligation to enumerate, per candidate, arm, matchup and
bank, both the **count and the rate** of produced final setups landing in
this set — and to use neither for selection, for any gate, or as grounds
for evaluation-time rejection sampling. That obligation lives in the
acceptance artifact rather than in a frozen contract, so it adds no design
decision and re-identifies no digest downstream agents already verify.

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
- **selector-audit randomness domain (review reconciliation)** — the contract says *100,000 draws per candidate x color x split ... resume must be exact set subtraction by draw id*.
  Agent 4's audit needs a selector seed per addressable draw id and the
  first freeze produced none — case_selector_seed covers only the 1,280
  bank-case seeds — so a tenth derived domain, selector_audit, was added
  under the existing selector_draw_seed root 2026081805. No root seed was
  added or changed and no threshold, candidate, bank, schedule or utility
  definition moved; it removes an unfrozen choice Agent 4 would otherwise
  have had to invent, and it moves exactly two contract digests
  (phase10_setup_contract_v1, phase10_setup_selector_v1) plus the bundle,
  leaving both bank digests, both bank manifests, the schedule digest, the
  scaler digest and the isolation-set digest byte-identical.
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
tests after    4879 passed, 3 skipped in 297.80s (0:04:57)
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

## 2. Agent 2 — Controlled Setup-Outcome Corpus

**Status: PASS** — 23/23 completion gates true, 16,384 games over 256 ordered family pairs at
64 each, zero utility fits, zero candidate selections, zero C1 optimizer steps,
and the accepted Phase 9 checkpoint byte-identical before and after.

Agent 2 creates outcome evidence and nothing else. It executes Agent 1's frozen
schedule, stores each game as a digest-checked record, seals the corpus, and
then proves by replay that the records say what happened.

### 2.1 Verified prerequisites

Every identity was recomputed from live bytes before a single game was played.

```text
Agent 1                         PASS, 23/23 gates, zero false gates
contract bundle digest          257f140dadddc00e4f75217ecedfe726390167de8769db0b5c40021e4388612f
outcome schedule digest         1a49f05032e300a8ecef81aa09776ed0d0766149576afb8eaa74a97e974e98b0
validation bank digest          a37ff113d03a0f67e760e447a462cc0d0d8de83f063d395715aeb77be355657f
test bank digest                be04b891ba5ab142aacbd937ab24f79054843310e8d28f6b8cbee65daef931ad
Phase 9 checkpoint SHA-256      dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
Phase 9 model-state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 9 parameters              863,959, all finite
Phase 7 library content         7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
Phase 7 splits                  6,400 / 800 / 800 at 400 / 50 / 50 per family
schedule audit                  11/11 checks, 16,384 games, 256 pairs, 64 each
test-bank neural/outcome access 0
```

Both evaluation banks were rebuilt only to re-derive their digests. No case was
played, scored, or shown to a model; the access log records both reads as
`digest_computation`.

### 2.2 Storage

```text
resolved root                   /Volumes/Brandon_Washington/stratego_phase10/corpus
resolution source               pointer_file
external volume                 /Volumes/Brandon_Washington, mounted, distinct device id from /
free bytes                      994,426,912,768
record bytes                    25,378,889
metadata bytes                  13,787,193
journal bytes                   7,885,090
total bytes                     47,051,172
bytes per game                  2872
payload compression ratio       0.356 (zlib level 6)
shards                          12
```

The root is a diagnostic, never an identity: corpus identity is the corpus
version, the logical game ids, and the payload/metadata/commit digests, so the
same bytes copied to another volume are the same corpus. A test copies a corpus
to a different path and re-derives the identical content digest.

### 2.3 The crash-safe commit protocol

`phase10_outcome_commit_v1` reproduces the accepted Phase 8 commit protocol for
a different payload. The rule is Phase 8's rule — a game becomes visible only
when its commit line exists — with the same write order and the same
truncation-based recovery:

```text
1. encode + compress + decode-verify the payload
2. build and check the metadata line
3. append the payload frame, flush
4. append the metadata line, flush
5. append the commit line, flush
```

Each commit carries the two file sizes after its own writes, which is what makes
recovery a truncation rather than a rewrite. Shards roll over only between games.

Crashes were injected at every stage before collection began, on a scratch store
that is deleted afterwards:

```text
before_payload       committed 3 of 6, discards the victim, 0 bytes discarded
after_payload        committed 3 of 6, discards the victim, 1605 bytes discarded
after_metadata       committed 3 of 6, discards the victim, 2447 bytes discarded
before_commit_flush  committed 3 of 6, discards the victim, 2447 bytes discarded
after_commit         committed 4 of 6, keeps the victim, 0 bytes discarded
shard_rollover       committed 3 of 6, discards the victim, 0 bytes discarded
```

```text
SIGKILL drill        worker killed (exit -9) with 2 of 6 committed;
                     recovery kept exactly those, resume under 3 workers
                     replayed exactly the 4 missing games
partition drill      the same games collected at worker_count 1 and 5 produce
                     the identical content digest ef851f84c46c871aa5b7cfc50105af0a...
```

The canonical corpus order is `sorted(game_id)` and nothing else, which is why
a differently partitioned run is the same corpus rather than a similar one.

### 2.4 The record

Agent 1 froze a 27-field outcome schema. Agent 2's own instruction
additionally requires a per-side trait-vector identity, the final setup
fingerprints, a record version and the contract/schedule digests, so a stored
record carries 37 fields of which the frozen 27 are a strict subset —
asserted at import time, not merely intended. Pre-game and post-game fields
are two closed, disjoint sets in the stored bytes:

```text
setup half     25 fields   identity, both sides' complete sampler provenance,
                          base ids, fingerprints, trait identities, seeds, digests
outcome half   9 fields    result, winner, red score, plies, decisions,
                          terminal reason, move-policy and checkpoint identity
derived         3 fields    payload / metadata / commit digests, which name bytes
                          that only exist once the record is written
```

A record carries no opponent-private value, no model score, no strength signal
and no physical path; a test greps the stored JSON for each.

### 2.5 Collection

```text
games                           16,384
plies                           5,658,357
workers                         12 pure-CPU processes, 1 torch thread each
wall clock                      1079 s
throughput                      15.19 games/s, 5246 decisions/s
peak worker RSS                 323,649,536 bytes
checkpoint loads                12 (one per long-lived worker owner)
inference failures              0
illegal neural actions          0
```

Both sides of every game play the accepted Phase 9 checkpoint under the frozen
behaviour — greedy, float32, `single_request`, no search, no temperature. The
accepted file is opened read-only; its weights are exported once to the frozen
evaluation format and the export is refused unless every tensor round-trips
bitwise, which is the accepted Phase 9 Agent 8 procedure unchanged.

### 2.6 Balance audit

```text
total games                     16,384
ordered pairs                   256
games per pair                  [64]
train split violations          0
duplicate game ids              0
duplicate commit identities     0
invalid setups                  0
stranded sampled setups         0
inventory violations            0
setup provenance mismatches     0
policy identity mismatches      0
non-finite inference rows       0
illegal neural actions          0
distinct base setups used       6,371
distinct final fingerprints     25,576
```

Every stored side was rebuilt from its provenance alone through
`rebuild_from_provenance`: 32,768 sides, zero mismatches.

**Diagnostics only** — these numbers rank nothing and select nothing:

```text
Red wins                        8,129  (0.496)
draws                           160  (0.010)
Red losses                      8,095  (0.494)
plies  min / mean / max         1 / 345 / 1623
```

Terminal reasons:

```text
battleless_move_limit_draw      160
flag_capture                    15,989
opponent_no_legal_move          235
```

Per-ordered-pair counts, Red scores, mean lengths and distinct base counts are
in `agent_02_family_pair_audit.csv`, one row per ordered pair.

### 2.7 Replay and the negative control

```text
games replayed end to end       16,384 of 16,384 (stride 1)
ordered pairs covered           256
families covered                16 of 16
W/D/L, length, terminal reason  identical on every replayed game
final setup fingerprints        identical on every replayed side
replay wall clock               1075 s
```

A replay audit that passes whichever weights played is not an audit, so the
same verifier was run against a deliberately wrong checkpoint — the accepted
Phase 8 anchor, a real and complete but different C1 model:

```text
sampled games                   64
games whose outcome differed    64 (1.000)
policy-identity check           fires: a worker loading the wrong weights is refused
result verifier                 fires: the stored outcomes are not reproduced
```

Device agreement was measured rather than assumed: on 32 games spread
across the corpus, cpu and mps chose identical games (zero disagreements).

### 2.8 Phase 9 preservation

```text
SHA-256 before                  dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
SHA-256 after                   dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model-state digest before       f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
model-state digest after        f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
parameters before / after       863,959 / 863,959
C1 optimizer steps              0
source opened                   read-only; weights exported, never rewritten
```

### 2.9 The seal

```text
state                           COLLECTING -> SEALED
committed games                 16,384
content digest                  1977bb6f5e2611b0498c7976f6129718fdfe7f6f44216f3b3f1932c8192b3c50
immutability                     a sealed corpus refuses every writer and every
                                truncation, including reconciliation
```

The content digest is taken over every committed payload digest in canonical
game-id order, so it is independent of worker count, segment, shard and path.

### 2.10 Recorded readings

- **stored record field count** — contract text: *Persist one digest-checked record per game containing at minimum: ...*.
  Agent 1 froze a 27-field schema; Agent 2's own minimum list additionally names a per-side trait-vector identity, the final setup fingerprints, a record version and the contract/schedule digests. The stored record is therefore the frozen 27 plus exactly those, 37 fields in all, and the store asserts the frozen 27 remain a strict subset.
  Safe because no frozen field changed meaning or left the record, the corpus contract document is untouched (its digest still recomputes to 951025f1...), and every added field is a structural descriptor or a digest — never an opponent-private value, a model score or a strength signal.

- **collection device** — contract text: *move behavior: greedy, float32, single_request, no search*.
  The frozen behaviour names no device, so the device is operational. CPU float32 with one thread per worker was chosen because it is roughly twice as fast as MPS at batch 1 on this 864k-parameter model and is bit-exact run to run.
  Safe because the device appears in no identity, and `device_agreement_probe` measures CPU-versus-MPS action agreement on a spread sample rather than assuming it.

- **move-policy identity prefix** — contract text: *move-policy identity*.
  The corpus policy reference is built with the accepted `neural_policy_ref` helper, whose naming convention prefixes every neural policy id with `phase6_`; the resulting token is `phase6_phase10_corpus_move_v1_greedy@0.2.0+float32`.
  Safe because hand-rolling a token would drop the helper's decision-rule and dtype versioning, which is the part of the identity that actually constrains replay; the prefix is the project's neural-policy family marker, not a phase claim.

- **negative control source** — contract text: *Run a wrong-checkpoint negative control*.
  The wrong checkpoint is the accepted Phase 8 anchor, a real and complete but different C1 checkpoint, rather than a perturbed copy of the accepted Phase 9 weights.
  Safe because writing a mutated copy of the artifact this phase must preserve byte for byte is a risk with no compensating benefit; a genuinely different checkpoint is a stronger control.

### 2.11 Evidence

```text
tests before   4879 passed, 3 skipped in 301.34s (0:05:01)
tests after    4964 passed, 3 skipped in 301.91s (0:05:01)
```

```text
reports/phase_10_data/agent_02_outcome_corpus.json
reports/phase_10_data/agent_02_family_pair_audit.csv
reports/phase_10_data/agent_02_acceptance.json
```

### 2.12 Handoff to Agent 3

Agent 3 fits exactly two utility models and makes no selection decision. It
receives a SEALED, read-only corpus of 16,384 records at content digest
`1977bb6f5e2611b0498c7976f6129718...`, reachable through
`phase10_storage.default_corpus_root` and `OutcomeReader` rather than a path;
the canonical record order `sorted(game_id)`; the exact schema and both halves'
field lists; the per-side setup descriptors, including base identity, family,
trait identity, complete sampler provenance and final fingerprints; the
train-only standardization source of 6,400 bases; and the proof that no
validation or test outcome was read, no held-out base entered the corpus, and
no Phase 9 weight moved.


## 3. Agent 3 — Utility Models and Independent Fit Audit

Status: **PASS** — 19/19 completion gates true.
Agent 3 fit exactly the two frozen utility models from the sealed
`phase10_setup_outcome_corpus_v1` (content digest `1977bb6f5e26...`),
audited the fit through an independent numpy path, and selected nothing:
both models go forward to Agent 4, and no validation or test outcome was
touched (neither bank stores one, and this agent played zero games).

### 3.1 Verified prerequisites

Agents 1 and 2 are PASS with no false gate. All eight contract digests, the
bundle (`257f140d...`), the outcome-schedule digest, both bank digests and
manifests, the Phase 9 isolation set, the Phase 7 library
(`7b8a6660...`, 6,400/800/800), and the accepted Phase 9 checkpoint (file
SHA `dfd698e5...`, model state `f1df694d...`, 863,959 parameters, all
finite) were recomputed from live bytes. The live
`phase10_setup_utility_v1` document equals Agent 1's frozen artifact copy
byte for byte, so no learning-design decision was left to make here. The
corpus was verified SEALED at its accepted content digest before fitting
and re-verified byte-identical after all work.

### 3.2 The fitting-input allowlist

The 37-field record is storage and provenance, not a feature set. Fitting
reads records only through `AllowlistedRecord`
(`stratego/training/phase10_utility_fit.py`), which raises on any field
outside the model's frozen allowlist:

```text
model_F: game_id, red_family, blue_family, result
model_T: + red_base_setup_id, blue_base_setup_id
```

`game_id` orders rows, `result` rebuilds the target through the frozen
mapping (red win 1.0, draw 0.5, red loss 0.0) — the stored `red_score` is
never read by fitting — and the base ids resolve each side's *base* through
`setup_library_v1` into the frozen `phase10_trait_feature_v1` 47-scalar
representation through the frozen train-only scaler. The other
31 stored fields (final fingerprints, provenance,
seeds, attempts, terminal reason, plies, decisions, digests, policy
identity, match seed, ordinal, winner, red_score, ...) are forbidden by
complement: accessing one is an exception, and the fields actually accessed
are recorded in the artifact (`accessed_fields`).

### 3.3 Feature reconstruction and the standardizer

Every one of the 8,000 library entries had its trait vector rebuilt from
its stored placement via `compute_trait_vector` and compared to the stored
vector: 0 mismatches. The 35-field to 47-scalar flattening was
re-derived independently from `TRAIT_SCHEMA` and its name order equals the
frozen feature names. The standardizer was recomputed with plain numpy over
**all 6,400 train bases** (`ddof=0`): mean and std match Agent 1's frozen
literals exactly, the production scaler digest is the frozen
`fa6eb1c1...`, and there are no zero-std fields. The corpus touches
6371 unique bases, all train-split; both recorded per-side trait
identity digests were re-derived and matched for every record
(32768 digests).

### 3.4 The two fits

Both models were fit exactly once, in canonical corpus order, from the
exact all-zero parameter vector, under the frozen protocol (CPU float64,
full-batch BCE + L2 1e-3 on raw family offsets and trait weights, intercept
unpenalized, L-BFGS lr 1.0, max 500 iterations, history 50, tolerance_grad
1e-10, tolerance_change 1e-12, strong Wolfe, single-threaded):

```text
model_F  objective 0.674125  bce 0.671843  l2 0.00228261
         iterations 15  evaluations 17  grad max 1.77e-07
model_T  objective 0.662444  bce 0.661085  l2 0.00135909
         iterations 148  evaluations 247  grad max 3.61e-07
```

The logit uses centered offsets while the penalty uses raw ones, so the
minimizer self-centers: observed raw-offset means are at the
2.8e-17 level. Objective values are diagnostics; they rank nothing.

Production coefficients and the scaler live in
`checkpoints/phase10/setup_utility_v1.json`
(SHA-256 `50cb947dae633417...`), referenced by digest from the
artifacts. Coefficient digests: model_F
`7bc2539af6045e47...`, model_T `d898782a2ae7cf4e...`.

### 3.5 Deterministic refit

Each model was refit 2 more times in independent processes from the same
all-zero initialisation. The frozen criterion — bit-exact equality of the
canonical coefficient JSON — held for every fit of both models
(max abs coefficient difference 0.0, objective spread 0.0, digests
identical across 3 processes per model).

### 3.6 The independent audit

The audit (`stratego/training/phase10_utility_audit.py`) rebuilt the design
without the production fit helper: placements -> trait vectors -> its own
flattening -> the frozen scaler literals -> per-record standardized
features, targets from the stored W/D/L token through its own mapping, and
Red/Blue orientation re-derived from each game id (families, ordinal, match
seed, winner/`red_score` consistency — all 16384 records, zero
violations). Production and audit designs agree exactly (targets, family
indices, game ids, and features to 0.0 max abs difference).

From the exported coefficients alone it recomputed logits, sigmoid
probabilities, BCE, L2, the full objective, the centering, and the analytic
gradient, all finite:

```text
model_F  |L_audit - L_reported| = 1.11e-16   grad max 1.77e-07   FD worst 2.68e-11
model_T  |L_audit - L_reported| = 0.00e+00   grad max 3.61e-07   FD worst 9.79e-11
```

All tolerances (objective/logit agreement 1e-10, stationarity 1e-6,
finite-difference 1e-6 + 1e-6|value|, centering 1e-8, refit exact) were
frozen in the audit module before any comparison ran.

### 3.7 Negative controls

| control | outcome |
| --- | --- |
| `orientation_swap` | fired |
| `orientation_swap_model_F` | fired |
| `wrong_draw_target` | fired |
| `held_out_scaler` | fired |
| `permuted_trait_column` | fired |
| `altered_family_id` | fired |
| `altered_coefficient` | fired |

Each control corrupts one thing the audit is supposed to catch — reversed
pair orientation (both models), draws scored 0.0, a validation-split
standardizer, a swapped trait column, a tampered family id, a tampered
coefficient — and every one was detected by the same checks that pass on
the true inputs.

### 3.8 Production-input safety

The exported artifact decomposes to own-side `u(s, c)`: closed key sets at
the root and per model, exactly 2 x 16 offsets and 2 x 47 weights indexed by
own colour / own family / own feature, no opponent-conditioned table, no
matchup matrix, no outcome-conditioned production feature. The scorer's
entire surface is `utility(model_id, colour, family_id, trait_vector)` —
there is no opponent argument to pass — and the recorded game logits equal
`intercept + u(red) - u(blue)` on sampled records to
2.2e-16. The red-first intercept is stored as a diagnostic and no
scoring path reads it.

### 3.9 No model selection

Model F and Model T were not compared by any strength signal. Both go
forward to Agent 4 with the six frozen candidate definitions. Corpus result
counts (8129 red wins / 160 draws / 8095 red losses) remain diagnostics.

### 3.10 Phase 9 and corpus preservation

The accepted Phase 9 checkpoint hashed identical before and after all Agent
3 work (file SHA and model-state digest; zero optimizer steps). The sealed
corpus re-verified at its accepted content digest with the seal intact:
Agent 3 opened no writer and reconciled nothing.

### 3.11 Recorded readings

Four readings are recorded in the acceptance artifact rather than decided
silently:

- **"fit exactly once" vs the deterministic-refit requirement** — one
  canonical fit per model produced the accepted coefficients; the two
  subprocess refits per model are byte-identical replays run only for the
  determinism gate, informed nothing, and were discarded.
- **single-threaded execution** — the frozen protocol names device and
  precision but not thread count; `torch.set_num_threads(1)` fixes the
  reduction order so the exact-equality refit criterion is meaningful, and
  is recorded in every fit's diagnostics.
- **the held-out-scaler control's inputs** — the control reads
  validation-split bases' *structural* trait vectors only (bytes the frozen
  bank construction already reads), proves the wrong scaler is detected,
  and discards it; no outcome exists or was read.
- **audit-internal gradient tolerances** — stationarity 1e-6 and
  finite-difference 1e-6 + 1e-6|value| were frozen in the audit module
  before any comparison ran.

One implementation finding is worth the reviewing chat's attention: the
frozen all-zero initialisation makes every logit exactly 0.0 at step 0,
where a hand-composed stable BCE (`clamp`/`abs`/`log1p` pieces) autodiffs
to a wrong subgradient (`-y` instead of `sigmoid(0) - y`), and L-BFGS then
line-searches a non-descent direction and terminates without moving. The
fit therefore computes BCE through
`torch.nn.functional.binary_cross_entropy_with_logits`, whose backward is
the analytic `sigmoid(eta) - y`, exact at 0; the unit suite pins the
analytic gradient at the zero start and the audit's finite-difference
checks confirm it at the fitted point.

### 3.12 Evidence

```text
tests before   4964 passed, 3 skipped in 301.74s (0:05:01)
tests after    5023 passed, 3 skipped in 303.94s (0:05:03)
```

Machine-readable: `reports/phase_10_data/agent_03_utility_models.json`,
`reports/phase_10_data/agent_03_utility_audit.json`,
`reports/phase_10_data/agent_03_acceptance.json`.

| gate | value |
| --- | --- |
| `agents1_2_pass` | true |
| `coefficients_finite` | true |
| `corpus_digest_verified` | true |
| `corpus_train_only` | true |
| `deterministic_refit_pass` | true |
| `full_suite_green` | true |
| `independent_objective_audit_pass` | true |
| `model_f_fit_complete` | true |
| `model_t_fit_complete` | true |
| `negative_controls_fire` | true |
| `no_candidate_selection` | true |
| `no_test_outcome_access` | true |
| `no_validation_outcome_access` | true |
| `objectives_finite` | true |
| `phase9_checkpoint_unchanged` | true |
| `production_scorer_own_side_only` | true |
| `red_blue_orientation_audit_pass` | true |
| `standardizer_train_only` | true |
| `trait_vectors_reconstructed` | true |

### 3.13 Handoff to Agent 4

The fitted `setup_utility_v1` artifact (path + SHA above), both coefficient
digests, the scaler digest `fa6eb1c1...`, the pure own-side scoring
contract, the six frozen candidates (P10-A..F over the 0.35/0.65 mixture),
proof that no held-out outcome was used (the corpus is train-only; neither
bank stores an outcome; zero games played), and proof Phase 9 is unchanged.
Agent 4 implements sampling and diversity only — the utility models are
frozen from here.

Two obligations carry forward: (1) Agent 4 must exhaustively
collision-check the materialized `selector_audit` seed universe when its
millions of draw ids exist — Agent 1's 58,792-seed audit does not cover
them; (2) Agent 2's 32-game CPU-vs-MPS probe is evidence about those games
only, never to be cited as exhaustive backend identity — the corpus is
authoritative as pure-CPU float32 evidence.
