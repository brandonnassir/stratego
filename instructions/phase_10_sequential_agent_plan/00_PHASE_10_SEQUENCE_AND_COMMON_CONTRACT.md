# Phase 10 — Sequence and Common Contract

## Mission

Phase 10 is a **learned setup-selection phase**, not another move-policy training phase. It asks whether game outcomes can be used to learn a better distribution over the frozen Phase 7 setup library while preserving setup diversity, information safety, reproducibility, and the accepted Phase 9 move model.

The intended system is:

```text
setup_library_v1
    -> lightweight setup utility
    -> setup_selector_v1
    -> frozen reflection/perturbation
    -> initial setup

+ accepted Phase 9 C1 move policy, unchanged
```

A full autoregressive setup Transformer is out of scope. Phase 11 remains belief validation. Phase 12 remains decision-time search.

## Sequential ownership

Agents work strictly in this order:

1. Agent 1 — contracts, seeds, banks, acceptance freeze
2. Agent 2 — controlled setup-outcome corpus
3. Agent 3 — utility models and independent fit audit
4. Agent 4 — selector and production setup source
5. Agent 5 — bounded validation selection
6. Agent 6 — integration soak and production freeze
7. Agent 7 — independent final acceptance and Phase 10 freeze

Each agent starts only after the reviewing chat accepts the previous agent. A `BLOCKED` or `FAIL` result is not permission for later agents to improvise around the contract.

## Frozen upstream identities

Every agent must verify these from live bytes/artifacts before doing its mission.

### Accepted Phase 9 move model

```text
path: checkpoints/phase9/selfplay_c1_v1.pt
SHA-256: dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model-state digest: f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
parameters: 863,959
C1 config digest: 31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d
```

The Phase 9 model must remain byte-identical throughout Phase 10.

### Frozen Phase 7 setup stack

```text
setup library: setup_library_v1
content digest: 7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
metadata digest: d86f486182a820d546d470ef4ebce92ff60c6259aed80c481bc985bce8c64980
manifest digest: 53139ab7e21c4e8a31987507d6fb1eabf93f36cdc1221fe85d08042963488f31
families: 16
bases: 8,000
train: 6,400 = 400/family
validation: 800 = 50/family
test: 800 = 50/family
profile baseline: neutral_v1
```

Also frozen: `setup_generator_contract_v1`, `setup_family_v1`, `setup_trait_vector_v1`, `setup_diversity_standard_v1`, `setup_perturbation_v1`, `setup_sampler_v1`, and `setup_source_v1`.

### Other frozen identities

```text
rules: stratego_project_v1
reference engine: phase2_1_reference_1.2.0
observation: observation_v2_1_127ch
engine action encoding: source_destination_10000_v1
model action frame: perspective_normalized_squares
model contract: model_contract_v2
trajectory: trajectory_v1
backend: KEEP_PYTHON
```

Agent 1 must also verify the full accepted Phase 9 contract/amendment chain from live bytes. Phase 10 does not rewrite Phase 9 history.

## Non-goals

No Phase 10 agent may:

- update C1 policy/value/belief weights;
- run PPO or continue Phase 9 RL;
- change the Phase 7 library, splits, family definitions, reflection, or perturbation semantics;
- use opponent true setup/family/base/seed or other hidden opponent information as selector input;
- change rules, observation channels, action encoding, or engine semantics;
- build a full setup Transformer;
- perform Phase 11 belief redesign or Phase 12 search;
- use human games;
- begin the official 168-hour campaign;
- use Phase 10 final-test outcomes for candidate selection or threshold changes.

Hard invariant:

```text
Phase 9 checkpoint before Phase 10 == Phase 9 checkpoint after Phase 10
```

Both file SHA and model-state digest must match.

## Phase 10 root seeds

Freeze these exact roots and derive all subordinate streams with domain-separated hashing; never use a process-global RNG cursor.

```text
master:                    2026081801
outcome-corpus schedule:   2026081802
setup draws:               2026081803
utility fitting:           2026081804
selector/candidate draws:  2026081805
validation cases:          2026081806
validation bootstrap:      2026081807
final-test bootstrap:      2026081808
```

## Controlled setup-outcome corpus

The only Phase 10 utility-training data is `phase10_setup_outcome_corpus_v1`.

Exact schedule:

```text
move policy both sides: accepted Phase 9 checkpoint
move behavior: greedy, float32, single_request, no search
setup split: Phase 7 train only
ordered family pairs: 16 x 16 = 256
games per ordered family pair: 64
total games: 16,384
```

Every record must carry enough setup provenance and outcome identity to replay independently. At minimum: game id, both base ids, both families, trait-vector identity, reflection/perturbation provenance, W/D/L, length, terminal reason, move-policy identity, setup seeds, and payload/metadata/commit digests.

## Utility models

Fit exactly two models.

### Model F — family-only

```text
u_F(s,c) = b[c, family(s)]
```

### Model T — family + traits

```text
u_T(s,c) = b[c, family(s)] + w[c]^T x(s)
```

`x(s)` is the frozen 35-field trait vector standardized using **only all 6,400 train bases** with population mean/std (`ddof=0`). A zero-std field is standardized to 0 and recorded.

Outcome target from Red perspective:

```text
Red win = 1.0
Draw = 0.5
Red loss = 0.0
```

Game logit:

```text
eta = red_first_intercept + u(red_setup, red) - u(blue_setup, blue)
p = sigmoid(eta)
```

The intercept is a fit diagnostic and is not used to rank setups. Center the 16 family offsets to zero mean per color for identifiability.

Frozen fit protocol:

```text
device: CPU
precision: float64
objective: full-batch BCE + L2 1e-3 on family/trait parameters
intercept penalty: none
optimizer: deterministic full-batch L-BFGS
max iterations: 500
history size: 50
gradient tolerance: 1e-10
change tolerance: 1e-12
line search: strong Wolfe if supported; otherwise Agent 1 must freeze one deterministic equivalent before outcomes exist
```

No hyperparameter search is permitted.

## Selector semantics

Allowed selector inputs are only:

```text
own color
requested Phase 7 split
candidate base's own family
candidate base's own trait vector
selector identity
selector seed
```

For utility `u`, temperature `T`:

```text
p_learned(s|c,split) = softmax(u(s,c)/T)
```

Final base distribution:

```text
p_phase10 = 0.35 * p_neutral_v1 + 0.65 * p_learned
```

After base selection, use the accepted Phase 7 path unchanged:

```text
reflection probability: 0.5
perturbation probability: 0.5
swap count: 1..6 uniform
Hamming distance: 2..12
retry semantics: unchanged
```

New identities should include `setup_utility_v1`, `setup_selector_v1`, and `learned_setup_source_v1`; do not redefine `neutral_v1` or `setup_source_v1`.

## Exactly six candidates

No seventh candidate.

```text
P10-A  family-only     T=0.75
P10-B  family-only     T=1.25
P10-C  family-only     T=2.00
P10-D  family+traits   T=0.75
P10-E  family+traits   T=1.25
P10-F  family+traits   T=2.00
```

All use the same 35% neutral / 65% learned mixture. The two utility models are fit once; candidate-specific refitting is forbidden. `neutral_v1` is a baseline, not a seventh candidate.

## Phase 10 evaluation banks

Agent 1 freezes both before utility fitting.

```text
phase10_validation_bank_v1
128 logical paired cases
Phase 7 validation split
8 cases per opponent-setup family

phase10_test_bank_v1
512 logical paired cases
Phase 7 test split
32 cases per opponent-setup family
```

Each case must include fixed held-out opponent setup provenance, selector draw seed(s), deterministic color pairing, match seed, family identity, and bootstrap unit.

Because earlier phases already used the Phase 7 held-out base universe, Phase 10 does **not** claim a wholly unseen base-template universe. It must instead guarantee:

```text
new logical case ids
new Phase 10 seeds
new procedural descendants
zero exact final-setup fingerprint overlap with Phase 9 validation/test cases
no Phase 9 per-case outcome used for Phase 10 fitting or selection
```

The test bank may be structurally built/audited before Agent 7, but no neural/game-outcome evaluation on it is allowed before Agent 7.

## Validation matchups

For every candidate, with the Phase 9 move model frozen:

1. learned selector vs neutral selector, same Phase 9 policy both sides;
2. Phase9+learned vs Strategic;
3. Phase9+learned vs Tactical;
4. Phase9+learned vs Phase 8 anchor;
5. Phase9+learned vs Random;
6. Phase9+learned vs Basic.

For matchups 2–6, also evaluate Phase9+`neutral_v1` on the exact same logical cases so setup-selector deltas can be computed.

All move play is greedy float32 `single_request`, no search.

## Candidate-selection score

```text
Delta_D = EWR(learned selector vs neutral selector) - 0.5
Delta_O = EWR(Phase9+learned vs O) - EWR(Phase9+neutral vs O)

S10 =
0.40*Delta_D
+ 0.30*Delta_Strategic
+ 0.20*Delta_Tactical
+ 0.10*Delta_Phase8
```

Random and Basic are validation guards, not score components.

Tie-break:

1. higher `S10`;
2. higher `Delta_Strategic`;
3. higher `Delta_D`;
4. higher normalized family entropy;
5. higher effective base diversity;
6. lexicographically smaller candidate id.

Validation point guards:

```text
Random overall EWR >= 0.95
Basic EWR >= 0.80
```

A candidate failing correctness/diversity/reproducibility is ineligible regardless of score. If no candidate is eligible, overall Phase 10 returns `FAIL` and later agents stop.

## Diversity contract

Apply to the **final mixed distribution**, for every candidate, color, and split:

```text
normalized family entropy H/log(16) >= 0.85
effective families exp(H) >= 10
minimum family probability >= 0.015
maximum family probability <= 0.18
within-family normalized base entropy >= 0.70 for every family
maximum conditional base probability within a family <= 0.10
```

Agent 4 additionally executes at least 100,000 complete selector draws **per candidate x color x split** through base selection, reflection, perturbation, and engine validation. Across these audits require zero illegal setups, inventory errors, stranded sampled setups, split violations, provenance mismatches, determinism mismatches, or non-finite selector values, and require all 16 families represented.

## Final acceptance gates

Agent 7 evaluates the single frozen winner on `phase10_test_bank_v1`.

### Gate A — direct learned-v-neutral non-inferiority

Ordinary PASS:

```text
EWR >= 0.49
paired 95% LB > 0.47
```

Direct improved criterion:

```text
EWR >= 0.52
paired 95% LB > 0.50
```

### Gate B — strong-opponent league non-inferiority

```text
Delta_L = 0.45*Delta_Strategic + 0.35*Delta_Tactical + 0.20*Delta_Phase8
```

Require:

```text
Delta_L >= -0.01
paired 95% CI lower bound > -0.03
```

Significant league improvement requires `Delta_L > 0` and CI lower bound `> 0`.

### Gate C — individual strong-opponent guards

For Strategic, Tactical, and Phase 8 anchor separately:

```text
paired selector-minus-neutral CI lower bound > -0.03
```

### Gate D — easy-opponent guards

```text
Random overall >= 0.95
Random Red >= 0.90
Random Blue >= 0.90
Basic >= 0.80
```

For Random and Basic also require paired learned-minus-neutral CI lower bound `> -0.03`.

### Gate E — diversity

Every diversity threshold above passes.

### Gate F — correctness and information safety

Zero illegal setups, inventory errors, stranded sampled setups, split leakage, provenance mismatch, hidden-opponent selector inputs, illegal neural moves, non-finite selector outputs, and inference failures.

### Gate G — reproducibility

```text
logical game id + selector seed + selector identity + requested split + color
-> same base -> same reflection -> same perturbation -> same final fingerprint
```

independent of worker order and process restart.

### Gate H — Phase 9 preservation

Exact accepted Phase 9 file SHA, model-state digest, parameter count, and zero C1 optimizer steps.

All eight are hard gates.

Final classification:

- `PASS-IMPROVED` only if all eight gates pass **and** Gate A improved criterion passes **and** Gate B is significantly positive;
- `PASS-NONINFERIOR` if all eight hard gates pass but the improved criteria are not both met;
- `FAIL` if the experiment runs correctly but a hard gate fails;
- `BLOCKED` if prerequisite identity/sealing/discipline evidence cannot be verified.

Report-only diagnostics never rescue a failed gate.

## Statistics

Use paired logical setup cases as bootstrap units.

```text
method: paired-unit percentile bootstrap
RNG: NumPy PCG64 or already-accepted deterministic project equivalent
replicates: 10,000
confidence: 95%
validation bootstrap root: 2026081807
final bootstrap root: 2026081808
```

Each matchup/difference receives a domain-separated token. Gate booleans and selection scores must be recomputed from primitive recorded outcomes.

## Contracts Agent 1 must freeze

```text
phase10_setup_contract_v1
phase10_setup_outcome_corpus_v1
phase10_setup_utility_v1
phase10_setup_selector_v1
phase10_selector_schedule_v1
phase10_eval_bank_v1
phase10_acceptance_v1
phase10_system_v1
```

`phase10_system_v1` binds the accepted Phase 9 checkpoint, accepted utility/scaler, selected selector config, and frozen Phase 7 reflection/perturbation path. The move model and selector remain separate artifacts.

## Storage/path semantics

Logical identities are path-independent. Use resolver/pointer semantics for Phase 10 data. Prefer the verified external volume for substantial corpus/replay bytes, but never hard-code `/Volumes/...` into logical scheduling/model/selector identity. If a pointer names an absent external volume, stop `BLOCKED`; do not silently create an internal replacement path.

## Reporting

Every agent appends a section to:

```text
reports/phase_10_implementation_report.md
```

Machine-readable artifacts live under:

```text
reports/phase_10_data/
```

Every acceptance artifact records status, gates, upstream identities, new digests, suite evidence, deviations, and handoff.

## Stop conditions

Return `BLOCKED` rather than improvising if an accepted upstream identity mismatches, held-out data enters fitting, selector inputs require opponent-private information, exact bank fingerprint isolation cannot be established, final-test outcomes are accessed before Agent 7, or storage identity/mount safety fails.

Return `FAIL` rather than redesigning if no candidate survives validation or final strength/diversity gates fail. Do not reopen Phase 9, change Phase 7, add candidates, or retune after results.

## Phase 11 handoff after PASS

Preserve permanently:

```text
neutral_v1
accepted learned_setup_source_v1
accepted Phase 10 selector config/utility
accepted Phase 9 selfplay_c1_v1.pt
```

Phase 11 validates the belief system; it does not retune setup selection.
