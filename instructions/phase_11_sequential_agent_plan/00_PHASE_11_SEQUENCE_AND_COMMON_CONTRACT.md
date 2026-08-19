# Phase 11 — Sequence and Common Contract

## Purpose

Phase 11 is a **belief-system validation and search-readiness phase**.

Scientific question:

> Does the existing belief head produce accurate, calibrated, information-safe, reproducible beliefs about hidden opponent ranks, and can those marginal beliefs be converted into complete legal hidden Stratego worlds quickly enough for Phase 12 search?

Phase 11 is **not** a training phase. It must not retrain, fine-tune, temperature-calibrate, replace, or otherwise modify the accepted neural model after examining Phase 11 evidence.

If the belief system fails, Phase 11 ends as `FAIL` and a separate repair phase must be designed. Do not turn this validation phase into a repair loop.

## Upstream freeze

Before Agent 1 begins, Phase 10 must have a formal closure commit. Agent 1 records that commit from live Git state; do not invent it from prose.

Permanent upstream inputs:

- Phase 9 model: `checkpoints/phase9/selfplay_c1_v1.pt`
  - SHA-256 `dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea`
  - model-state digest `f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd`
  - parameters `863,959`
  - C1 config digest `31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d`
- Phase 10 selector:
  - P10-D
  - `model_T`
  - T = `0.75`
  - mixture = `0.35 neutral_v1 / 0.65 learned`
  - config SHA-256 `6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668`
- Phase 10 utility coefficient digest `d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc`
- Phase 10 scaler digest `fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9`
- Phase 10 production-system digest `615cc3c3a4fab6e4400e20a5a93b13a08c43ab6c3ca63828c6a64742e98175d2`
- Phase 7 setup library, `neutral_v1`, accepted observation/model/event contracts remain frozen.

The Phase 9 checkpoint already contains the belief head. Phase 11 validates that head; it does not create a replacement neural belief model.

Agent 1 must derive and freeze a **belief-head tensor identity** from live checkpoint tensor names + bytes.

## Absolute prohibitions

No Phase 11 agent may:

1. update any Phase 9 neural parameter;
2. run a belief optimizer step;
3. calibrate or temperature-scale the belief head;
4. change the 127-channel observation design;
5. change P10-D, utility, scaler, temperature, or mixture;
6. modify Phase 7 setup generation;
7. begin Phase 12 search;
8. use Phase 11 test evidence to repair or retune anything;
9. use hidden opponent truth as an input to belief inference or sampling;
10. silently change a frozen metric, threshold, bank, seed, or statistical procedure.

The evaluator may read true hidden ranks **only after predictions are produced**, on a privileged scoring path isolated from production inference.

## Belief target

For each opponent piece whose true rank is not publicly known, predict a distribution over the 12 Stratego ranks.

Publicly known ranks are not ordinary hidden-rank examples and must not inflate hidden-rank accuracy.

## Baselines

Exactly two:

### `remaining_count_belief_v1`

Marginal probabilities from publicly inferable remaining rank inventory, obeying public impossibility constraints such as moved pieces not being Bomb/Flag.

Primary predictive baseline.

### `count_uniform_world_sampler_v1`

A nonlearned complete-world sampler from the remaining legal piece multiset.

Search fallback / joint-sampling baseline only.

Neither baseline may read hidden truth.

## Phase 11 banks

Freeze both before any Phase 11 prediction evidence exists.

### `phase11_validation_bank_v1`

- 512 logical paired cases
- 2 games/case
- 1,024 games total
- 64 cases per opponent-behavior stratum
- within each stratum: 32 opponent P10-D setup-source cases, 32 `neutral_v1`
- observer is Red once and Blue once per case

### `phase11_test_bank_v1`

- 2,048 logical paired cases
- 2 games/case
- 4,096 games total
- 256 cases per stratum
- within each stratum: 128 P10-D setup-source cases, 128 `neutral_v1`
- observer is Red once and Blue once per case

### Eight opponent-behavior strata

1. accepted Phase 9 policy
2. Phase 8 anchor
3. Strategic rule opponent
4. Tactical rule opponent
5. Basic rule opponent
6. `information_miser`
7. `scout_rush`
8. `miner_rush`

Observer side is always the accepted Phase 9 policy/belief model. Opponent side is the named stratum. Observer own setup source is frozen by Agent 1 and must be constant across both banks; default P10-D unless a verified implementation constraint prevents it.

Game outcomes are report-only and rank nothing.

## Prediction record

Minimum logical fields:

- bank version
- logical case id
- game id
- decision index
- acting perspective/color
- opponent stratum
- opponent setup-source stratum
- public-state identity
- hidden-piece public tracker identity
- public square/location
- public legal-rank mask
- learned probability vector
- remaining-count baseline probability vector
- true rank, privileged evaluator only
- progress bucket
- moved/unmoved public status
- model/prediction identity
- observation/public-history identity

## Statistics

Predictions within a game are correlated. Do **not** bootstrap pieces independently.

Final CIs:

1. aggregate metric contributions within each logical paired case;
2. keep both color games together;
3. bootstrap logical cases;
4. 10,000 percentile-bootstrap replicates, 95%.

Stratum CIs bootstrap cases within stratum.

Primary metrics:

- hidden-piece CE/NLL
- top-1 rank accuracy
- Brier score
- ECE
- probability of true rank
- entropy

Mandatory diagnostic slices:

- opponent stratum
- Red/Blue observer perspective
- early/middle/late
- moved/unmoved
- rank
- opponent setup source

`R_CE = CE_learned / CE_remaining_count`

## `belief_sampler_v1`

Convert marginals to a complete legal hidden-rank assignment.

Frozen algorithm:

1. lock publicly known ranks;
2. compute exact remaining inventory;
3. apply public impossibility masks;
4. derive deterministic random unresolved-piece order from sample identity;
5. process pieces in that order;
6. for each piece, legal ranks = public legal ranks with remaining count > 0;
7. weight each legal rank by `learned_probability * remaining_count`;
8. renormalize and sample via deterministic domain-separated stream;
9. decrement chosen count;
10. if all legal weights are zero, fallback for that step to normalized remaining counts;
11. continue until complete;
12. verify exact inventory and every public fact.

The sample seed determines piece order and categorical draws.

## Hard world constraints

Every sampled world must obey:

- exact remaining counts
- known/revealed ranks
- captures/dead pieces
- moved piece cannot be Bomb/Flag
- public Scout deductions
- ownership
- alive/dead status
- public start information
- known rank-by-start information
- all other frozen public deductions

One invalid sampled world fails correctness.

## Information-safety attack

At least **50,000 hidden-truth permutation trials**.

For each trial, alter hidden truth while holding public state/history identical, then run belief inference and a fixed-seed sample.

Required exactly zero:

- belief-output differences
- fixed-seed sample differences
- forbidden hidden-input accesses

## Large sampler audit

At least **250,000 complete worlds** across thousands of distinct validation public states.

Zero-tolerance counters:

- inventory errors
- public-knowledge violations
- known-rank violations
- immobility violations
- impossible assignments
- nonfinite rows
- provenance mismatches
- hidden-input accesses

World-diversity statistics are report-only unless Agent 1 freezes a threshold before sampling exists.

## Reproducibility

A sampled world is a pure function of:

- public-state identity
- belief-model identity
- sampler identity
- sample ordinal/seed

It must not depend on worker count, call order, process id, path, wall clock, or previous calls.

Required topology/restart coverage:

- 1 worker
- 4 workers
- 12 workers
- forward order
- reverse order
- fresh process
- kill/resume or exact set-subtraction resume

## Runtime benchmark

Measure:

- one belief forward
- forward + 16 worlds
- forward + 32 worlds
- forward + 64 worlds

Hard readiness ceiling:

> p95(one belief forward + 64 complete legal worlds) <= 500 ms

Record model-forward and sampling times separately.

## Final hard gates

### Gate A — predictive superiority

- `R_CE <= 0.97`
- paired 95% upper bound for `CE_learned - CE_baseline` `< 0`

### Gate B — top-1 improvement

- `Delta_top1 >= +0.03`
- paired 95% lower bound `> 0`

### Gate C — calibration

- overall ECE `<= 0.08`
- no opponent stratum ECE `> 0.12`
- learned-minus-baseline Brier 95% upper bound `<= +0.01`

### Gate D — robustness

Every opponent stratum: `R_CE <= 1.05`

### Gate E — sampler correctness

All correctness counters zero.

### Gate F — information safety

All hidden-truth/output/sample/input-access counters zero.

### Gate G — reproducibility and runtime

All deterministic topology/restart comparisons exact and p95 forward+64 <= 500 ms.

### Gate H — preservation

After Phase 11:

- exact Phase 9 SHA/state
- parameter count 863,959
- C1 optimizer steps 0
- exact belief-head identity
- exact P10-D config
- exact Phase 10 utility/scaler
- exact Phase 7 library

## Final classification

Exactly one:

- `PASS-SEARCH-READY` — all A-H pass
- `FAIL` — experiment valid but >=1 gate fails
- `BLOCKED` — integrity/sealing/prerequisites cannot be established

If `FAIL`, Phase 12 is not authorized; design a separate belief-repair phase.

## Phase 11 identities

Agent 1 freezes at least:

- `phase11_belief_contract_v1`
- `phase11_belief_baseline_v1`
- `phase11_belief_bank_v1`
- `phase11_belief_metrics_v1`
- `phase11_belief_sampler_v1`
- `phase11_information_safety_v1`
- `phase11_acceptance_v1`
- `phase11_system_v1`

## Root seeds

Freeze exactly:

- master `2026081901`
- bank/case schedule `2026081902`
- game/match randomness `2026081903`
- belief/world sampling `2026081904`
- information-safety trials `2026081905`
- reproducibility/runtime audit `2026081906`
- validation bootstrap `2026081907`
- final-test bootstrap `2026081908`

All random needs use named domain-separated derivations. No downstream agent invents a new randomness domain without reviewer authorization.

## Sequential execution

1. Agent 1 — contracts, seeds, banks, metrics, gates
2. Agent 2 — evaluator, baselines, validation predictive evidence
3. Agent 3 — constrained sampler + large audit
4. Agent 4 — information safety, reproducibility, runtime
5. Agent 5 — integrated validation and implementation freeze
6. Agent 6 — production soak and `phase11_system_v1` freeze
7. Agent 7 — first sealed test evaluation and final acceptance

A later agent starts only after reviewing-chat acceptance of the previous agent.

## Reporting

Append sections to `reports/phase_11_implementation_report.md`.

Machine-readable evidence under `reports/phase_11_data/`.

Every agent reports:

- starting revision
- ending revision if committed
- identities recomputed from live bytes
- completion-gate table
- suite before/after
- recorded deviations/readings
- forbidden-operation counters
- exact handoff identities

Paths/volumes are diagnostics, never logical identity.

## Commit discipline

Agents commit only after reviewer acceptance unless their instruction explicitly requires a pre-run administrative freeze.

Never rerun a sealed final evaluation merely to create a commit.

Agent 7 final work remains uncommitted until the reviewing chat accepts/rejects its recommendation.
