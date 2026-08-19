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

## 4. Agent 4 — Selector and Production Setup Source

Status: **PASS** — 24/24 completion gates true.
Agent 4 builds the setup source and proves it. It fits nothing, plays no
game, computes no strength signal and selects no candidate: all six
candidates go forward to Agent 5 exactly as Agent 1 froze them.

### 4.1 Verified prerequisites

Every identity was recomputed from live bytes before a distribution existed.

```text
Agents 1, 2, 3                  all PASS, zero false completion gates
contract bundle digest          257f140dadddc00e4f75217ecedfe726390167de8769db0b5c40021e4388612f
setup_utility_v1 file SHA-256   50cb947dae633417858dc3352ee1e68e41c1c54845c5d3a261f735571983c25d
model_F coefficient digest      7bc2539af6045e478cd3dbbf78e16c6123616d285a3f32dd1b1a5c1da96ad935
model_T coefficient digest      d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc
trait scaler digest             fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9
Phase 9 checkpoint SHA-256      dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
Phase 9 model-state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 7 library content         7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
splits                          6400 / 800 / 800
neutral_v1                      reflection 0.5, perturbation 0.5, uniform 1..6
sealed corpus                   1977bb6f5e2611b0498c7976f6129718fdfe7f6f44216f3b3f1932c8192b3c50 (0 records read)
bank neural/outcome access      0
```

Both coefficient digests are **recomputed from the live coefficients**, not
read back from the artifact's own stored labels, and the production file —
gitignored by Agent 1's policy — is compared field for field against the
tracked Agent 3 record it was reviewed as. A matching label on tampered
bytes therefore cannot pass.

Three prerequisites are checked mechanically rather than declared:

```text
six candidates          match Agent 1's frozen matrix: True, and Agent 3's handoff: True
                        6 distinct selector identities
test-bank outcome access Agents 1/2/3/4 all zero: True
plays no game           the selector module imports no neural framework, checkpoint reader,
                        Phase 9 module, evaluation harness or match runner (AST check): True
```

`no_strength_selection_games` is a claim about code, so it is checked
against the code: the selector *cannot* produce a strength signal because
it imports nothing that could, whatever any record asserts.

### 4.2 The selector

A selector call reads six things and nothing else: own colour, requested
split, selector identity, selector seed, and the candidate base's own
family and own trait vector. Utility is consumed only through Agent 3's
accepted own-side scorer, whose entire surface is
`utility(model_id, colour, family_id, trait_vector)` — there is no opponent
argument to pass, no centering re-derived by hand, and no path in this
agent reads the fitted Red-first intercept.

```text
branch      u < 0.35 -> neutral_v1 branch, else the learned branch
neutral     the base the accepted setup_sampler_v1 would have taken for
            (split, selector_seed, profile='neutral_v1')
learned     inverse-CDF walk over the split's bases in ascending
            (family_index, base_index), on float64 cumulative mass
then        the accepted Phase 7 path unchanged: reflection coin,
            perturbation coin, uniform swap count 1..6, frozen retry
            rules, and the complete final-output validation stack
```

Six decisions draw from six domain-separated streams and no mutable global
RNG cursor exists, so worker count, shard boundaries, call order and
process restarts cannot move a single draw.

The learned branch changes exactly one thing — which base is chosen. The
reflection coin, the perturbation coin and the swap count are the accepted
sampler's for that draw identity on both branches, which is what keeps the
frozen post-selection marginals intact when a learned base is substituted.

### 4.3 The 36 exact distributions

Every candidate x colour x split distribution is exact arithmetic over the
whole split, never an empirical frequency. All 36 are finite, non-negative,
sum to 1 within 1e-12, and reproduce `0.35*p_neutral + 0.65*p_learned`
**bit for bit** rather than to a tolerance.

Worst case over all 36 cells, against the frozen thresholds:

| metric | threshold | worst observed | pass |
| --- | --- | --- | --- |
| normalized family entropy | >= 0.85 | 0.9817 | yes |
| effective families | >= 10.0 | 15.207 | yes |
| min family probability | >= 0.015 | 0.0442 | yes |
| max family probability | <= 0.18 | 0.1321 | yes |
| within-family base entropy | >= 0.7 | 0.9937 | yes |
| max conditional base probability | <= 0.1 | 0.03562 | yes |

All 6 candidates are diversity-eligible.
The 0.35 uniform component alone puts a floor of 0.35/16 = 0.021875 under
every family probability, so the minimum-family-probability threshold
cannot fail by construction; the other five are properties of the fitted
utility at each temperature.

Per-cell probability-vector digests are in `agent_04_selector_contract.json`
and the raw per-family and per-base metrics in `agent_04_diversity_audit.json`.

### 4.4 The seed universe

Agent 3 carried forward an explicit obligation: Agent 1's 58,792-seed audit
proved the *frozen* id space collision-free, but the millions of
`selector_audit` draw ids did not exist then. They exist now, so every one
was enumerated, together with the two production streams it feeds.

```text
agent1_frozen_universe         58,792 seeds      58,792 distinct
selector_audit              3,600,000 seeds   3,600,000 distinct
selector_base               3,600,000 seeds   3,600,000 distinct
selector_branch             3,600,000 seeds   3,600,000 distinct
combined                   10,858,792 seeds  10,858,792 distinct
collisions                 0
```

### 4.5 Review reconciliation: a real defect, found and fixed

The Agent 4 review challenged the reported worst empirical-vs-exact family
total variation of 0.04332. The challenge was correct, and
reconciling it exposed an implementation defect. The arithmetic is why:
on 100,000 draws over 16 families at p ~ 1/16, sampling noise
predicts a TV near 0.00482, so the
reported value was roughly nine times too large to be noise. The signature
across cells said the same thing: TV tracked temperature, worst for the
sharpest candidates (T=0.75) and best for the flattest (T=2.00).

**Both sides classify the same concept.** The empirical counter reads
`SelectorDraw.family_id`, which *is* `base_entry.family_id` — the selected
base's frozen Phase 7 primary family, read off the library entry and never
recomputed from a descendant — and the exact vector is `p_mixed` summed over
the family blocks of the same frozen base order. Reflection and perturbation
are family-preserving and a descendant inherits `primary_family_id` verbatim,
so no post-reflection or post-perturbation concept enters either side. No
renaming was warranted; the gap had to be a defect, and it was.

**The defect.** The learned branch walked `cumsum(p_mixed)` where the frozen
contract says *cumulative softmax mass*. The branch coin has already applied
the 0.35 neutral weight before that ladder is reached, so walking the mixed
vector applied it a second time and realized

```text
realized   0.35*neutral + 0.65*(0.35*neutral + 0.65*learned)
         = 0.5775*neutral + 0.4225*learned
intended   0.3500*neutral + 0.6500*learned
```

The empirical distribution was therefore pulled toward uniform.

Replaying the 100,000 frozen draw ids of
P10-D blue validation from their `selector_audit`, `selector_branch`
and `selector_base` seeds, through an inverse-CDF written independently of
the selector, settles it. Both ladders are shown on the same draw ids, so
the challenged number can be read straight off the table:

| family | exact | empirical (fixed) | count | residual | empirical (defective) | residual |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | 0.132081 | 0.134150 | 13,415 | +0.002069 | 0.109890 | -0.022191 |
| F01 | 0.083803 | 0.084670 | 8,467 | +0.000867 | 0.076960 | -0.006843 |
| F02 | 0.073830 | 0.073360 | 7,336 | -0.000470 | 0.069400 | -0.004430 |
| F03 | 0.050158 | 0.049970 | 4,997 | -0.000188 | 0.054930 | +0.004772 |
| F04 | 0.055429 | 0.055680 | 5,568 | +0.000251 | 0.058000 | +0.002571 |
| F05 | 0.056811 | 0.057020 | 5,702 | +0.000209 | 0.058380 | +0.001569 |
| F06 | 0.044492 | 0.043820 | 4,382 | -0.000672 | 0.050880 | +0.006388 |
| F07 | 0.044986 | 0.044520 | 4,452 | -0.000466 | 0.050400 | +0.005414 |
| F08 | 0.057315 | 0.057960 | 5,796 | +0.000645 | 0.059550 | +0.002235 |
| F09 | 0.046507 | 0.046940 | 4,694 | +0.000433 | 0.052050 | +0.005543 |
| F10 | 0.056057 | 0.055040 | 5,504 | -0.001017 | 0.057800 | +0.001743 |
| F11 | 0.053849 | 0.053390 | 5,339 | -0.000459 | 0.056310 | +0.002461 |
| F12 | 0.055395 | 0.056070 | 5,607 | +0.000675 | 0.058710 | +0.003315 |
| F13 | 0.053060 | 0.053400 | 5,340 | +0.000340 | 0.056730 | +0.003670 |
| F14 | 0.085754 | 0.084020 | 8,402 | -0.001734 | 0.075900 | -0.009854 |
| F15 | 0.050474 | 0.049990 | 4,999 | -0.000484 | 0.054110 | +0.003636 |

The defective column is the signature: the largest families (F00 at
0.1321 exact) are systematically
under-drawn and the smallest systematically over-drawn, every residual
pointing toward uniform. The fixed column shows no such structure — its
residuals are the two-sided scatter of ordinary sampling noise.

```text
TV, contract ladder (cumsum p_learned)   0.00549
TV, defective ladder (cumsum p_mixed)    0.04332
sampling-noise expectation               0.00482
reproduces the challenged 0.04332          True
independent replay vs production         0 disagreements
```

The defective ladder reproduces the challenged number exactly, and the
contract ladder lands at the noise floor. The fix is one line of meaning:
`cumulative_learned` is `cumsum(p_learned)` and is the only ladder the
learned branch walks.

**What did not change.** No candidate, utility coefficient, temperature,
mixture weight, threshold, seed derivation or evaluation bank was touched.
The exact `p_neutral`, `p_learned` and `p_mixed` vectors and every published
probability-vector digest are identical before and after — the exact
distributions were always right, and so were the diversity conclusions drawn
from them. What was wrong was that the sampler did not realize them. Every
sampling, diversity and reproducibility result below was regenerated after
the fix.

No acceptance threshold on empirical total variation was introduced: it
remains a diagnostic. The fix is pinned instead by an exact structural unit
test — `cumulative_learned == cumsum(p_learned)`, with a discriminating
assertion that the two ladders genuinely differ — plus a realized-mixture
test, so this cannot regress silently.

### 4.6 The large sampling audit

```text
draws per candidate x colour x split   100,000
cells                                  36
total complete selector draws          3,600,000
workers                                12
wall clock                             9.1 min
throughput                             6594 draws/s
```

Every draw went selector -> base -> reflection -> perturbation -> the
accepted engine validation stack, was rebuilt from its own recorded
provenance, and was compared against the accepted Phase 7 sampler for the
same draw identity — a neutral-branch draw field for field, a learned-branch
draw on every base-independent decision.

```text
determinism_mismatches           0
illegal_setups                   0
inventory_errors                 0
non_finite_selector_values       0
provenance_mismatches            0
split_violations                 0
stranded_sampled_setups          0
all 16 families, every cell      True
```

**Diagnostics only** — these rank nothing and select nothing:

```text
empirical-vs-exact family total variation   worst 0.00708 over 36 cells
empirical-vs-exact base total variation     worst 0.10202
neutral-branch frequency                    0.3467 .. 0.3529   (frozen weight 0.35)
reflection frequency                        0.4964 .. 0.5037   (frozen 0.5)
perturbation-requested frequency            0.4961 .. 0.5021   (frozen 0.5)
Phase 9 held-out fingerprint landings       207,211 of 3,600,000 draws
  train                                           0 / 1,200,000 = 0.0000
  validation                                 45,865 / 1,200,000 = 0.0382
  test                                      161,346 / 1,200,000 = 0.1345
```

That last diagnostic is the residual Agent 1 recorded and deliberately
left unrejected, so it is worth reading rather than skipping. Train lands
**zero** times, which is the sanity check: the Phase 9 held-out universe is
drawn from the validation and test splits, so a train draw cannot land in
it. The validation and test rates independently corroborate Agent 1's
rejection walk, which fired on 7 of 256 validation and 141 of 1,024 test
selector seeds — 2.7% and 13.8% against the 3.8% and 13.4% measured here.
The mechanism is the same one Agent 1 named: the unperturbed branch
reproduces a held-out base template exactly, and roughly half of all draws
are unperturbed.

This is a **report-only diagnostic and never a gate**. Rejecting such draws
at draw time would distort the very mixed distribution the diversity
contract is stated over, which is precisely why Agent 1 forbade it. Base-id
reuse across phases is allowed; what Phase 10 guarantees is that the setups
a *case* fixes carry no exact Phase 9 fingerprint overlap, and that no
Phase 9 per-case outcome informs any Phase 10 fit or selection.

### 4.7 Topology, restart and resume

```text
fixed draw-id set             18,000 ids across all 36 cells
worker counts                 1, 3, 8, 13
orderings                     contiguous, round-robin, reversed
configurations                12, all identical to the reference
fresh process                 identical: True
resume                        exact set subtraction by draw id; 3,600 recomputed, all identical
```

A replay's record is the canonical digest of the whole draw — base id,
reflection, perturbation identity, final fingerprint and complete
provenance — so 'identical' is the whole object, not a sampled field.

### 4.8 The permitted-input boundary

`SelectorRequest` carries exactly three fields — split, colour and selector
seed — and refuses to be built from a mapping that carries anything else.
All 16 positive controls were rejected:

```text
opponent_family, opponent_base_setup_id, opponent_setup_fingerprint, opponent_final_setup, opponent_seed, opponent_policy_id, opponent_checkpoint, game_outcome
result, red_score, winner, matchup_token, match_seed, game_id, hidden_opponent_truth, storage_path
```

Varying hidden opponent truth across 16 whole opponent
contexts left every draw bit-identical, no public selector API takes an
opponent-shaped parameter, and a produced record carries no opponent,
outcome, winner, Elo or reward field.

### 4.9 Preservation

```text
Phase 9 SHA-256 before / after   dfd698e5b6cf536a523bdd35dd3a32a5... / unchanged
Phase 9 model-state before/after f1df694d59e3435994be06f2537d9c60... / unchanged
Phase 9 parameters               863,959
C1 optimizer steps               0
Agent 2 corpus                   SEALED, digest unchanged, 0 records read
Agent 3 utility + scaler         byte-identical, 0 refits
neutral_v1                       consumed, never redefined
```

### 4.10 Recorded readings

- **After base selection, delegate to the accepted Phase 7 reflection/perturbation implementation un** — the selector re-derives the accepted setup_sampler_v1 decision streams through the public derive_stream_seed under the accepted neutral_v1 profile object — reading that profile's own reflection probability, perturbation probability and intensity weights rather than restating them — and then calls the accepted build_descendant, the sampler's single construction path. No Phase 7 byte is touched. The adapter's identity is proven, not asserted: every one of the audited neutral-branch draws is compared field for field against sample_setup(split, seed, 'neutral_v1') and every learned-branch draw is required to share that baseline's reflection, perturbation coin and swap count, differing in the base alone
- **setup_sampler_v1 provenance field `sampler_profile`** — a learned-branch draw records sampler_profile='neutral_v1' because that field names the frozen post-selection profile actually used (reflection 0.5, perturbation 0.5, uniform 1..6), which is true on both branches and is what makes a neutral-branch draw bit-identical to the baseline. It says nothing about base selection: the branch, the candidate and the selector identity live in the Phase 10 selector provenance beside it, so no consumer has to infer the arm from a Phase 7 field
- **no_test_outcome_access** — the diversity contract is stated over all three splits, so the audit draws from the Phase 7 test *split*. That is structural sampling of base templates and is not access to phase10_test_bank_v1: no bank case was played, scored or shown to a model, the only bank reads are the two structural digest recomputations in the access log, and no outcome exists on either bank to read
- **at least 100,000 draws per candidate x color x split** — exactly 100,000 per cell, 3,600,000 in total, each carrying the full verification burden — construction through the accepted validation stack, a rebuild from its own provenance, and the accepted-sampler cross-check — rather than a larger count with a lighter check

### 4.11 Evidence

```text
tests before   5023 passed, 3 skipped in 303.94s (0:05:03)
tests after    5080 passed, 3 skipped in 314.99s (0:05:14)
```

```text
reports/phase_10_data/agent_04_selector_contract.json
reports/phase_10_data/agent_04_diversity_audit.json
reports/phase_10_data/agent_04_acceptance.json
```

| gate | value |
| --- | --- |
| `agents1_3_pass` | true |
| `all_16_families_represented` | true |
| `all_diversity_thresholds_recorded` | true |
| `candidate_count_exactly_six` | true |
| `distribution_diversity_audit_complete` | true |
| `full_suite_green` | true |
| `illegal_setups_zero` | true |
| `inventory_violations_zero` | true |
| `mixture_35_65_exact` | true |
| `neutral_v1_unchanged` | true |
| `no_strength_selection_games` | true |
| `no_test_outcome_access` | true |
| `opponent_hidden_inputs_rejected` | true |
| `phase9_checkpoint_unchanged` | true |
| `probabilities_finite` | true |
| `probabilities_sum_to_one` | true |
| `provenance_mismatches_zero` | true |
| `seed_collision_audit_clean` | true |
| `selector_contract_frozen` | true |
| `selector_draws_ge_required` | true |
| `split_violations_zero` | true |
| `stranded_sampled_setups_zero` | true |
| `topology_reproducibility_pass` | true |
| `utility_digests_match` | true |

### 4.12 Handoff to Agent 5

Agent 5 receives the six immutable selector configurations and their
probability-vector digests, the deterministic selector API and the
`neutral_v1` baseline API, the diversity eligibility of every candidate
(all six eligible), the validation bank identity, and the frozen score and
tie-break rule. It runs bounded validation selection on
`phase10_validation_bank_v1` only: `phase10_test_bank_v1` stays sealed until
Agent 7, and no corpus outcome may select. The two utility models and the
six temperatures are frozen — no refit, no retune, no seventh candidate.

## 5. Agent 5 — Bounded Validation Selection

Status: **PASS** — 17/17 completion gates true.
Agent 5 evaluates exactly the six frozen candidates on the validation
bank, applies the predeclared eligibility rules, and freezes one selector
configuration. It fits nothing, refits nothing, changes no temperature or
mixture weight, adds no candidate, and never opens the test bank.

### 5.1 Verified prerequisites

Every identity was recomputed from live bytes before a game existed.

```text
Agents 1, 2, 3, 4               all PASS, zero false completion gates
contract bundle digest          257f140dadddc00e4f75217ecedfe726390167de8769db0b5c40021e4388612f
setup_utility_v1 file SHA-256   50cb947dae633417858dc3352ee1e68e41c1c54845c5d3a261f735571983c25d
model_F coefficient digest      7bc2539af6045e478cd3dbbf78e16c6123616d285a3f32dd1b1a5c1da96ad935
model_T coefficient digest      d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc
trait scaler digest             fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9
selector contract digest        ed1198f3a4bfc8f73264cf22602f6d8ba89d9458e9ae5c8a8ddf7f0543e35e59
Phase 9 checkpoint SHA-256      dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
Phase 9 model-state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 9 parameters              863,959
Phase 8 anchor export SHA-256   cd0b22d24d36dbe01f88897c3e2bde325b7e141d07d092edc74918e6b0cd6dda
Phase 7 library content         7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
validation bank digest          a37ff113d03a0f67e760e447a462cc0d0d8de83f063d395715aeb77be355657f
validation manifest digest      459cef36d7032beb8fc9665efa7692dac3c40c68109e9f0bcdefa6141bd0906e
test bank digest                be04b891ba5ab142aacbd937ab24f79054843310e8d28f6b8cbee65daef931ad (structural only)
sealed corpus                   1977bb6f5e2611b0498c7976f6129718fdfe7f6f44216f3b3f1932c8192b3c50 (0 records read)
neutral_v1                      reflection 0.5, perturbation 0.5, uniform 1..6
```

All 36 published probability-vector digests — six candidates x two
colours x three splits — were **rebuilt from the coefficients and
compared**, not read back from Agent 4's record, and every rebuilt
distribution was required to satisfy `p_mixed == 0.35*p_neutral +
0.65*p_learned` bitwise.

Agent 5 began from commit `e1df780`. The intermediate `258644e` carried
the defective sampler; no sampling evidence produced under it was read,
and none is admissible for candidate selection.

### 5.2 The learned branch, re-derived before the first game

The Agent 4 review found that the learned branch had walked
`cumsum(p_mixed)`, double-applying the 0.35 neutral weight. That fix is
upstream of everything Agent 5 measures, so Agent 5 re-derives it
independently rather than inheriting the claim — three readings, none of
them a re-run of Agent 4's own assertions, all completed **before any
validation game was played**.

Structural, by parsing the production source:

```text
branch-coin calls in draw()      1
base-uniform calls in draw()     1
mixture-weight comparisons       ['branch_uniform < NEUTRAL_MIXTURE_WEIGHT']
bare 0.35/0.65 literals          none
attributes the walk reads        ['base_count', 'cumulative_learned', 'searchsorted']
ladder construction              cumulative_learned = np.cumsum(p_learned)
```

The 0.35/0.65 choice occurs exactly once, as `branch_uniform <
NEUTRAL_MIXTURE_WEIGHT`, at the branch decision. The inverse-CDF walk
reads `cumulative_learned` and `base_count` and nothing else; `p_mixed`
is not in its reachable set, so the defect cannot recur by editing a
constant.

Exact, over all 36 candidate x colour x split cells: the ladder was
recomputed from `p_learned` alone and matched **bitwise** in every cell;
it differs from `cumsum(p_mixed)` in every cell, so the check
discriminates; the inverse-CDF interval widths reproduce `p_learned` to
5.5e-17; and `0.35*p_neutral + 0.65*p_learned` equals the published
`p_mixed` bitwise.

Runtime, over frozen draws:

```text
draws                            2,000
branch-coin calls                2,000  (one per draw: true)
base-uniform calls               1,283  (learned-branch draws: 1,283)
```

#### The structural negative control

A check that only ever passes proves nothing about its own sensitivity.
A shadow walk over `cumsum(p_mixed)` was therefore run on the *identical*
branch coins and base uniforms, and is required to visibly reproduce the
superseded behaviour. In closed form the shadow realizes
`0.5775*neutral + 0.4225*learned` rather than the frozen 0.35/0.65.

```text
cell                    production TV   p_mixed-ladder TV   TV to the 0.5775/0.4225 blend
P10-A/red                   0.003897           0.037118                     0.004307
P10-A/blue                  0.005376           0.039858                     0.004967
P10-D/red                   0.004319           0.039300                     0.003285
P10-D/blue                  0.004845           0.042351                     0.004157
```

Production sits at the sampling-noise scale against the published
distribution; the shadow ladder sits roughly an order of magnitude away
from it and at the noise scale against the double-mixed blend. The
P10-D/blue row reproduces the value the Agent 4 review challenged
(0.04332 reported, 0.043318 recomputed there). These total variations are
diagnostics: they add no acceptance threshold, and the fix is pinned by an
exact structural test, not by a statistical one.

### 5.3 What was played

```text
bank                    phase10_validation_bank_v1, 128 logical paired cases
selector seat           accepted Phase 9 checkpoint, in all six matchups
opposing seat           the Phase 9 checkpoint in learned_vs_neutral; otherwise
                        the matchup's own opponent
move behaviour          greedy, float32, single_request, no search
colour pairing          the evaluated selector plays Red in game 0, Blue in game 1
bootstrap unit          the logical case, scoring the mean of its two games
learned arm             6 candidates x 6 matchups x 128 cases x 2 games
neutral arm             5 matchups x 128 cases x 2 games, on the identical cases
games                   10,496 in 317s on 12 workers (cpu)
inference failures      0
```

A case fixes the held-out opponent setup, the two selector draw seeds and
the two `neutral_v1` own-side draws, so the learned arm and the baseline
arm differ in exactly one thing: which base the selector chose. Every
candidate saw the same 128 cases, the same opponent setups and the same
bootstrap units; only the cell identity — arm, candidate, matchup — is
candidate-specific, and it is carried in `match_id` so no two cells can
share a cached game.

`learned_vs_neutral` has two sides and two selectors, so the held-out
opponent setup has no seat in it: the neutral side plays the case's frozen
`neutral_v1` draw for the colour it was dealt. The other five matchups seat
that held-out setup opposite the selector under test, identically in both
arms.

The rule-based opponents play on the frozen
`case_match_seed(case_id, game_index, matchup)`, which is independent of arm
and candidate exactly as Agent 1 required, so Strategic, Tactical, Random
and Basic draw identical randomness in both arms. The accepted runner
derives a side's seed from `match_id`, which here must stay
candidate-specific, so the two requirements are met on different objects:
identity through `match_id`, randomness through a thin delegating wrapper
that replaces only the request's two seed fields. Handed the runner's own
seed the wrapper is a no-op — 12 control games compared bit-identical on
every recorded field — and the selector-under-test side is never wrapped,
because greedy neural play reads no seed at all.

#### Seat reconciliation

The move policy is not the same on both seats of every game, and an earlier
draft of this section said it was. Rather than reword it and move on, the
claim was reconciled against the recorded games exhaustively: for all 10,496
of them the intended match specification was rebuilt and its identifier
compared with the recorded one. `match_id` is a blake2b hash over the whole
specification, both policy tokens included, so this is a cryptographic seat
check rather than a re-read of a stored label — a game played with a
different policy on either seat could not carry the identifier it carries.

```text
matchup              games   seat policy                                        role        red  blue  total
learned_vs_neutral    1536   phase6_phase10_eval_move_v1_greedy@0.2.0+float32   opposing    768   768   1536
learned_vs_neutral    1536   phase6_phase10_eval_move_v1_greedy@0.2.0+float32   selector    768   768   1536
vs_strategic          1792   phase6_phase10_eval_move_v1_greedy@0.2.0+float32   selector    896   896   1792
vs_strategic          1792   strategic_rule_based@1.1.0                         opposing    896   896   1792
vs_tactical           1792   phase6_phase10_eval_move_v1_greedy@0.2.0+float32   selector    896   896   1792
vs_tactical           1792   tactical_rule_based@1.0.0                          opposing    896   896   1792
vs_phase8_anchor      1792   phase6_c1_warmstart_greedy@0.2.0+float32           opposing    896   896   1792
vs_phase8_anchor      1792   phase6_phase10_eval_move_v1_greedy@0.2.0+float32   selector    896   896   1792
vs_random             1792   phase6_phase10_eval_move_v1_greedy@0.2.0+float32   selector    896   896   1792
vs_random             1792   random_legal@1.0.0                                 opposing    896   896   1792
vs_basic              1792   basic_heuristic@1.0.0                              opposing    896   896   1792
vs_basic              1792   phase6_phase10_eval_move_v1_greedy@0.2.0+float32   selector    896   896   1792

aggregate seats                                    observed  expected
basic_heuristic@1.0.0                                  1792      1792   match
phase6_c1_warmstart_greedy@0.2.0+float32               1792      1792   match
phase6_phase10_eval_move_v1_greedy@0.2.0+float32      12032     12032   match
random_legal@1.0.0                                     1792      1792   match
strategic_rule_based@1.1.0                             1792      1792   match
tactical_rule_based@1.0.0                              1792      1792   match
```

20,992 seats over 10,496 games, **0 mismatches** — in the seat policy, the
cell token, the frozen match seed and the colour pairing alike. Every
matchup is exactly half Red and half Blue on both seats. The aggregate
counts are derived from the frozen mapping rather than read off the data,
and they agree: the selector seat is the Phase 9 checkpoint in all six
matchups, which with the direct matchup's second Phase 9 seat gives 12,032,
and each of the five external opponents holds one seat in six candidate
cells plus the baseline cell, giving 1,792 each.

A token proves which policy a seat named; it does not prove which checkpoint
answered for it. That was checked separately, with a control in both
directions:

```text
matchup              seat under test                                    bound          reproduces  swap changes
learned_vs_neutral   phase6_phase10_eval_move_v1_greedy@0.2.0+float32   phase9              12/12         12/12
vs_phase8_anchor     phase6_c1_warmstart_greedy@0.2.0+float32           phase8_anchor       12/12         12/12
```

Replayed under the checkpoint the harness bound to that seat, every sampled
game reproduces its recorded replay digest; replayed with the other
checkpoint behind the same token, every one of them changes. The second half
is what makes the first half worth anything.

The audit reran no scheduled game (48 replays for the control, none
recorded) and changed nothing about the frozen selection.

### 5.4 The fixed neutral baseline

`neutral_v1` is the baseline, never a seventh candidate. Its own-side draws
were rebuilt live through the accepted Phase 7 sampler and required to
fingerprint exactly as Agent 1 froze them, so a moved sampler would have
stopped the run rather than quietly shifting every delta.

```text
matchup              EWR      W /  D /  L
vs Strategic         0.8516   218 /  0 /  38
vs Tactical          0.8047   206 /  0 /  50
vs Phase 8 anchor    0.8438   216 /  0 /  40
vs Random            0.9922   254 /  0 /   2
vs Basic             0.8613   220 /  1 /  35
```

These sit where the accepted Phase 9 evaluation put this checkpoint
(Random 0.9883, Basic 0.8535 on the Phase 9 test bank), which is the
cheapest available evidence that the harness reproduces the accepted move
model rather than a degraded copy of it. A stronger check was run directly:
a game played through this harness and through Agent 2's accepted collector
path on the same two setups produced an **identical action history**.

Sharding does not enter a result either: one recorded work unit (P10-D
vs_strategic, cases 32-48, 32 games) was deleted and rebuilt by a fresh
process running 1 worker instead of 12, and every recorded field came back
identical, digest included.

### 5.5 Candidate results

```text
id      model    T      S10      Delta_D  Delta_St  Delta_Ta  Delta_P8   Random   Basic
P10-A  model_F  0.75  +0.02891  +0.04492  -0.00781  +0.05469  +0.02344   0.9785  0.8652
P10-B  model_F  1.25  +0.03008  +0.05469  +0.01172  +0.02344  +0.00000   0.9844  0.8613
P10-C  model_F  2.00  -0.01328  -0.00391  -0.01953  -0.01562  -0.02734   0.9844  0.8301
P10-D  model_T  0.75  +0.04023  +0.05664  +0.02344  +0.04297  +0.01953   0.9883  0.8965
P10-E  model_T  1.25  -0.00586  -0.00586  -0.01758  +0.01172  -0.00586   0.9961  0.8438
P10-F  model_T  2.00  +0.02070  +0.03125  -0.00391  +0.03516  +0.02344   0.9922  0.8398
```

`Delta_D` is the direct learned-vs-neutral EWR minus 0.5; each `Delta_O` is
the learned-minus-neutral EWR difference on the same cases. Random and Basic
are guards and never score components.

Per-matchup EWR of every candidate, learned arm:

```text
id      direct   vs Strat  vs Tact   vs P8     vs Rand   vs Basic
P10-A  0.5449    0.8438    0.8594    0.8672    0.9785    0.8652   
P10-B  0.5547    0.8633    0.8281    0.8438    0.9844    0.8613   
P10-C  0.4961    0.8320    0.7891    0.8164    0.9844    0.8301   
P10-D  0.5566    0.8750    0.8477    0.8633    0.9883    0.8965   
P10-E  0.4941    0.8340    0.8164    0.8379    0.9961    0.8438   
P10-F  0.5312    0.8477    0.8398    0.8672    0.9922    0.8398   
```

### 5.6 Eligibility

A candidate is eligible only if Agent 4's correctness, reproducibility and
diversity all pass, validation Random EWR >= 0.95, validation Basic EWR >=
0.80, and every correctness counter is zero. A high score cannot rescue an
ineligible candidate.

```text
id      diversity   Random >= 0.95   Basic >= 0.80   correctness   eligible
P10-A  True        True             True            True          True
P10-B  True        True             True            True          True
P10-C  True        True             True            True          True
P10-D  True        True             True            True          True
P10-E  True        True             True            True          True
P10-F  True        True             True            True          True
```

All 6 of 6 candidates are eligible. Across 10,496 games the
zero-tolerance counters are all zero: no illegal setup, no illegal action,
no engine rejection, no policy exception, no contract violation, no
non-finite score, no inference failure and no unscored game.

### 5.7 The winner

```text
ranking          P10-D > P10-B > P10-A > P10-F > P10-E > P10-C
winner           P10-D  (model_T, T=0.75)
S10              +0.040234
  0.40*Delta_D          +0.022656   (Delta_D +0.056641)
  0.30*Delta_Strategic  +0.007031   (Delta_S +0.023438)
  0.20*Delta_Tactical   +0.008594   (Delta_T +0.042969)
  0.10*Delta_Phase8     +0.001953   (Delta_8 +0.019531)
tie-break        decided at level 1 (higher S10)
```

The score was recomputed from the primitive per-case game scores
independently of the helper that produced it, and the two agree to within
1e-15 for every candidate. The ranking was reproduced by the frozen
tie-break key and matches. No tie reached the candidate-id level.

P10-D is the family+traits model at the lowest temperature, and the only one
of the six whose four score components are all strictly positive.

Concentrating the distribution is what a low temperature does, and it is
exactly what the diversity contract exists to bound — so it is worth saying
plainly that P10-D is also the tightest candidate in the field: lowest
normalized family entropy (0.9817) and lowest effective base diversity
(748.4) of the six, and the worst cell in Agent 4's entire 36-cell audit. It
clears the 0.85 entropy floor and the 10-family effective count with wide
margin anyway: the frozen 35% uniform component puts a floor under
concentration that no temperature in the candidate matrix can reach.

The frozen configuration is `phase10_selector_config_v1`, written to
`reports/phase_10_data/agent_05_frozen_selector_config.json`. The selector
config and the utility coefficients remain separate artifacts, and no C1
checkpoint was created or altered.

### 5.8 Phase 9 fingerprint landings — report-only

Agent 1's standing obligation, carried forward by Agent 4, at the
granularity Agent 5 owes: candidate x arm x matchup x bank, count and rate.
A learned selector's own-side draw could not be enumerated when the banks
were built, and rejecting such a draw at evaluation time would distort the
very mixed distribution the diversity contract is stated over — so Agent 1
forbade rejecting it and required recording it instead.

```text
candidate    arm       per matchup   total          rate
P10-A        learned     5 / 256       30 /  1536   0.0195
P10-B        learned     4 / 256       24 /  1536   0.0156
P10-C        learned     6 / 256       36 /  1536   0.0234
P10-D        learned     7 / 256       42 /  1536   0.0273
P10-E        learned     2 / 256       12 /  1536   0.0078
P10-F        learned     9 / 256       54 /  1536   0.0352
neutral_v1   neutral     0 / 256        0 /  1280   0.0000
```

The per-matchup count is constant within a candidate because an own-side
draw depends on the case, the colour and the candidate — never on the
opponent — so the same setups play in all six matchups. The rates sit in
the band Agent 4 predicted from 3.6M audit draws (0.0381 on the validation
split). The baseline arm records zero by construction: Agent 1's rejection
walk already excluded the frozen `neutral_v1` own-side draws from the
Phase 9 held-out set.

**These values changed nothing.** They gate nothing, triggered no retry,
entered no score, no eligibility test and no tie-break.

### 5.9 Access discipline and Phase 9 preservation

```text
validation-bank game outcomes read   10,496
test-bank game outcomes read         0
test-bank neural inference           0
test-bank access                     structural digest recomputation only
utility models fit                   0
candidates added                     0
temperature / mixture changes        0 / 0
rescue reruns                        0
corpus records read                  0
human games used                     0
C1 optimizer steps                   0
```

The Phase 9 checkpoint was hashed before the work and again after it:

```text
before   dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
after    dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
state    f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd (unchanged: true)
```

### 5.10 Recorded readings

- **one held-out opponent setup ... plays in every matchup and in both arms** — the held-out opponent setup seats opposite the selector under test in the five externally-opposed matchups, in both arms and for all six candidates. It has no seat in learned_vs_neutral, which has two sides and two selectors: the learned draw plays the neutral_v1 draw of the colour the other seat was dealt, which is exactly the pair of neutral own-side draws Agent 1 froze per case. A third setup cannot enter a two-sided game.
- **match_seeds: one seed per (case, game index, matchup), independent of arm and candidate, so a rule-based opponent draws identical randomness in both arms** — the accepted runner derives a side's seed from match_id, and Agent 5 must also keep game identities candidate-specific, so the two requirements are met on different objects: match_id carries the cell (arm, candidate, matchup) through MatchSpec.setup_bank_version, while the opponent actually plays on case_match_seed(case_id, game_index, matchup) through a thin FrozenSeedPolicy wrapper that replaces only the request's two seed fields. The selector-under-test side is the accepted Phase 9 checkpoint playing greedy in all six matchups and reads no seed at all.
- **Stress, if run, is report-only** — no stress evaluation was run. Agent 5's mission is bounded to the six frozen candidates on the validation bank, and a report-only diagnostic cannot change a selection, so running one would add cost and no evidence.
- **full_suite_green** — the gate is a claim about a suite that contains the test asserting it, so a single run cannot evidence it: a false gate fails the suite, which keeps the gate false. The measurement therefore lives in its own recorded stage (`--record-suite`), the artifact test checks that the gate agrees with that measurement rather than asserting it directly, and the recorded run is the one taken with the artifact in its final state. The confirming run is reported alongside it.
- **Record every validation-bank game-outcome access** — the access log records one entry per stage and bank rather than one per game; the per-game count is carried alongside it as discipline.validation_bank_outcome_access, so the number of outcome reads is exact and the log stays readable.

### 5.11 Artifacts and completion gates

```text
reports/phase_10_data/agent_05_candidate_results.csv
reports/phase_10_data/agent_05_frozen_selector_config.json
reports/phase_10_data/agent_05_acceptance.json
```

Full suite: `.venv/bin/python -m pytest tests -q` — 5132 passed, 3 skipped in 315.48s (0:05:15).

`full_suite_green` is a claim about a suite that contains the test asserting
it, so one run cannot evidence it — a false gate would fail the suite and
keep the gate false. The measurement is recorded in its own stage, the
artifact test checks the gate against that measurement instead of asserting
it, and the run above was taken with the artifact in its final state. A
confirming run reproduced it exactly.

| gate | value |
| --- | --- |
| `agents1_4_pass` | true |
| `candidate_count_6` | true |
| `eligibility_rules_exact` | true |
| `frozen_selector_config_complete` | true |
| `full_suite_green` | true |
| `learned_branch_independently_verified` | true |
| `neutral_baseline_fixed` | true |
| `no_final_test_outcome_access` | true |
| `no_seventh_candidate` | true |
| `phase9_checkpoint_unchanged` | true |
| `same_cases_across_candidates` | true |
| `score_recomputes_exactly` | true |
| `tie_break_recomputes_exactly` | true |
| `unregistered_candidates_zero` | true |
| `utility_models_not_refit` | true |
| `validation_bank_identity_verified` | true |
| `winner_unique_or_tiebreak_resolved` | true |

### 5.12 Handoff to Agent 6

Agent 6 receives the frozen `phase10_selector_config_v1` (P10-D), its
SHA-256 `6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668`, the
unchanged utility artifact and trait scaler, the winner's train-split
production distribution digests, the `neutral_v1` baseline identity, the
accepted Phase 9 identity, and the complete validation evidence. Selection
is closed: Agent 6 may not reopen it, add a candidate, refit a utility or
change the 0.35/0.65 mixture.

`phase10_test_bank_v1` remains sealed. Its digest
`be04b891ba5ab142aacbd937ab24f79054843310e8d28f6b8cbee65daef931ad` was recomputed structurally and
matches; zero games, zero neural inferences and zero outcome reads touched
it. Agent 7 owns the first final-test evaluation.

