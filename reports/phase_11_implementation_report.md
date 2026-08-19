# Phase 11 Implementation Report

Phase 11 is a **belief-system validation and search-readiness** phase. It
asks whether the accepted Phase 9 belief head produces accurate,
calibrated, information-safe, reproducible beliefs about hidden opponent
ranks, and whether those marginals convert into complete legal hidden
worlds fast enough for Phase 12 search. Nothing is trained, calibrated or
repaired: `checkpoints/phase9/selfplay_c1_v1.pt` must be byte-identical
before and after the phase, and a failing system ends the phase as FAIL
rather than becoming a repair loop.

## 1. Agent 1 — Contracts, Seeds, Banks, Metrics, and Acceptance Freeze

**Status: PASS** — 31/31 completion gates true, zero problems, zero Phase 11 predictions, zero sampled worlds, zero games,
zero optimizer steps.

Agent 1 freezes the entire Phase 11 experiment before any prediction
score, sampler output or test outcome exists. Nothing below was chosen
after seeing a Phase 11 result, because no Phase 11 result exists yet.

### 1.1 Verified upstream identities

Every identity was recomputed from live bytes, not read from a record.

```text
Phase 10 closure commit         17188a5 (Agents 1-7 all PASS, Agent 7 PASS-NONINFERIOR)
Phase 9 checkpoint SHA-256      dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
Phase 9 model-state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 9 parameters              863,959, all finite; global optimizer step 47,086
belief head (live tensors)      belief_output.bias, belief_output.weight
belief-head digest              a9df48a1adcd29b1a46c42ff1e605ede485119a36c247f1ae74f249f6d6f1dc7
C1 config digest                31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d
P10-D config SHA-256            6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668
utility model_T digest          d898782a2ae7cf4ed1cb2833fad6e53d8407ec2048dafbd34a6a20c1c9766edc
trait scaler digest             fa6eb1c112defc4c1034831b84db8848181e1f674f8439c9c265916d89e8b7f9
phase10_system_v1 digest        615cc3c3a4fab6e4400e20a5a93b13a08c43ab6c3ca63828c6a64742e98175d2
Phase 8 anchor export           cd0b22d24d36dbe01f88897c3e2bde325b7e141d07d092edc74918e6b0cd6dda
Phase 7 library content         7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
Phase 7 splits                  6,400 / 800 / 800; neutral_v1 0.5/0.5
observation metadata digest     91cd890b8cd35be7aa8ddc18c44cae3669e27695c29d5f97f8816971676d8c15 (127 channels)
pre-existing Phase 11 work      none: no predictions, no worlds, no outcomes
```

The belief-head tensor identity is derived from the live checkpoint —
`belief_output.bias` and `belief_output.weight` hashed under the accepted
state-digest recipe — and frozen for every later agent to re-derive.

### 1.2 Frozen contracts

Eight documents, canonical JSON, SHA-256.

```text
phase11_belief_contract_v1        13c4607619cca5fde621980b6ffa155d3c01378f000d5869319a069286186f75
phase11_belief_baseline_v1        c017d51f78e8f7f5976abec62aa259fd8918508810c42bc41e88470c5998c197
phase11_belief_bank_v1            874a2513427aebde69c9d31c9d06c6562d3daf2faa3f0dc4cd916656a175516d
phase11_belief_metrics_v1         a2f7e5b4cc3944194d3b735f96ad805413670648ffd46acaac3a6a7f436368cb
phase11_belief_sampler_v1         a113d2e9588a6c4d7c2dcff954773e693ae876d19465904e4b277e86675afca9
phase11_information_safety_v1     1b8160d544b5ee71eb1b03be025a868e7298ace1d61c486524e631ba68faab4d
phase11_acceptance_v1             0121ecaac6849a59d78798833ec419f9ff12c14f8720f1bef259960f42c01fe5
phase11_system_v1                 9aa22d45ab85b65d5ed14e40288ef7cd4c3226e8d66f52508d9717929ac1adfe
bundle                            ad16f921c602c1e1eb4975bee31fa6d1dff8dd4afdd09c332d9deaa94712192d
```

`phase11_system_v1` binds what exists now — the accepted belief model
and its head identity, P10-D with utility/scaler, the Phase 7 library,
the baselines and the bank versions — and leaves five slots unbound with
their filling rules (evaluator, sampler implementation, safety evidence,
runtime benchmark result, bank digests). Agent 6 fills them at the
production freeze.

### 1.3 Seeds and derivations

```text
master                    2026081901     bank/case schedule        2026081902
game/match randomness     2026081903     belief/world sampling     2026081904
information safety        2026081905     repro/runtime audit       2026081906
validation bootstrap      2026081907     final-test bootstrap      2026081908
```

Beneath the eight roots sit **twelve derived domains** under the new
`strat-b11` personalization — disjoint from every accepted upstream tag:

```text
bank_observer_setup  bank_opponent_setup  bank_match       world_sample
world_order          world_categorical    safety_trial     repro_schedule
benchmark            bootstrap            soak_setup       soak_match
```

The Agent 6 soak namespace is frozen *now* — id formats, volumes
(1,024 games x 8 requests = 8,192), colour parity, request attachment
rule — closing the Phase 10 soak-namespace deviation in advance.

The collision audit enumerated 171,212 seeds across every currently enumerable id space and found 171,212
distinct values — zero duplicates inside a stream and zero collisions
across streams. The million-scale world-sample space is keyed by
public-state identities that do not exist yet; Agents 3, 4, 6 and 7
carry the frozen obligation to re-run this audit over every world
stream they actually derive.

### 1.4 Frozen banks

```text
phase11_validation_bank_v1   512 cases  1,024 games  validation split
phase11_test_bank_v1       2,048 cases  4,096 games  test split
8 opponent strata x 2 setup sources x 32/128 cases per cell
observer: accepted Phase 9 policy+belief head, P10-D setups, both banks
```

```text
validation bank digest    bba6860549c05ebd59487d83d205e9d18b2109ab143d3816afbe793a13a04023
validation manifest       d83ab48516e03a74695a04d68dcda6f17fbf02cb468b6785a3d91627b0534173
test bank digest          566ac35214ac04d5928af2f2975308a03bb78eb2a19e2ea05e6367f839eff404
test manifest             360a687d5a6ed2623d50a88cb1fe392dee85064f15f84fc61f13752b6ddca3b0
```

A case fixes both seats' setups (each drawn from its frozen source
conditioned on its own colour — never mirrored) and one match seed per
game. There is **no rejection of any kind**: Phase 11 selects nothing,
so every arrangement is exactly what production would produce. The
structural audits rebuild provenance, re-derive every draw
independently, validate every arrangement through the engine, and
rebuild sampled cases in isolation — all exact, for both banks; the
cross-bank check proves zero id, seed and fingerprint overlap.

P10-D branch mixture over the materialized draws:

```text
validation  learned   987 / neutral   549  (0.643 / 0.357 of 1536)
test        learned  4038 / neutral  2106  (0.657 / 0.343 of 6144)
```

### 1.5 Frozen target, metrics, sampler, safety and gates

- **Targets**: every live opponent piece not legally known to the
  observer, at every observer-acting decision — the engine's accepted
  `belief_target` semantics; publicly known ranks are never events.
  Rank order is the engine enumeration (spy..marshal, flag, bomb).
- **Learned vector**: raw float64 softmax of the head's 12 logits at
  the piece's square — no masking, no epsilon; CE floors only inside
  the log at 1e-12 (report-only counter).
- **Baseline**: `remaining_count_belief_v1`, mask-restricted
  count-proportional; provably positive on the true rank.
- **Sampler**: the common-contract twelve steps with
  `weight = learned_probability x remaining_count`, plus a frozen
  completion-feasibility guard on step 6's legal set (recorded reading:
  the unguarded walk can dead-end on feasible instances; the guard is
  exact and keeps every valid world reachable).
- **Statistics**: case-level percentile bootstrap, 10,000 replicates,
  95%, both colour games kept together, domain-separated streams per
  metric; ECE 15 equal-width bins, pooled events.
- **Gates A-H** exactly as the common contract, with explicit
  strict/non-strict operators and boundary tests in the suite;
  classification PASS-SEARCH-READY / FAIL / BLOCKED recomputes from
  gate booleans with no discretionary override.
- **Runtime**: backend frozen before measurement — CPU float32, one
  torch thread, 480 benchmark states over 48 cells; hard ceiling
  p95(forward + 64 worlds) <= 500 ms.

### 1.6 Access ledger and readings

The append-only ledger at `reports/phase_11_data/
phase11_bank_access_ledger.jsonl` records every Agent 1 bank access:
structural build, digest, audit and artifact write for each bank — all
structural-only with zero neural/scored/privileged/outcome counters.

11 recorded readings (bank split binding, colour-pairing order, no-rejection
draws, the sampler completion-feasibility rule, the soak namespace,
the shared bank-schedule root, progress-bucket thresholds, the
unit-uniform tail edge, hash-order schedule selection, and the
untracked Phase 10B drafts) are itemized in the acceptance artifact
for reviewer acceptance at this handoff.

### 1.7 Artifacts and completion gates

```text
reports/phase_11_data/agent_01_phase11_contract.json
reports/phase_11_data/agent_01_validation_bank.json
reports/phase_11_data/agent_01_test_bank.json
reports/phase_11_data/agent_01_acceptance.json
reports/phase_11_data/phase11_bank_access_ledger.jsonl
```

Full suite: `.venv/bin/python -m pytest tests -q` — 5280 passed, 3 skipped in 313.30s (0:05:13)

| gate | value |
| --- | --- |
| `acceptance_gates_frozen` | true |
| `bank_overlap_zero` | true |
| `baselines_frozen` | true |
| `belief_head_identity_frozen` | true |
| `bootstrap_frozen` | true |
| `classification_frozen` | true |
| `contract_bundle_frozen` | true |
| `eight_contracts_frozen` | true |
| `eight_root_seeds_frozen` | true |
| `full_suite_green` | true |
| `isolated_case_rebuild_pass` | true |
| `ledger_initialized` | true |
| `metrics_frozen` | true |
| `no_neural_updates` | true |
| `no_phase11_predictions_scored` | true |
| `observation_contract_verified` | true |
| `phase10_selector_identity_verified` | true |
| `phase7_identity_verified` | true |
| `phase8_anchor_identity_verified` | true |
| `phase9_checkpoint_unchanged` | true |
| `phase9_identity_verified` | true |
| `prediction_target_contract_frozen` | true |
| `randomness_domains_frozen` | true |
| `sampler_math_frozen` | true |
| `seed_collision_audit_clean` | true |
| `test_balance_exact` | true |
| `test_bank_exact` | true |
| `test_outcome_access_zero` | true |
| `upstream_phase10_closed` | true |
| `validation_balance_exact` | true |
| `validation_bank_exact` | true |

Agent 1 stops here and waits for reviewer acceptance. Agent 2 is
authorized for validation-bank predictive evaluation only; the test
bank stays sealed with zero scored access, proven through the ledger.

## 2. Agent 2 — Belief Evaluator, Baselines, and Validation Predictive Evidence

**Status: PASS** — 24/24 completion gates true, 1,024 validation games, 2,850,966 hidden-piece prediction events, zero optimizer steps, zero scored test-bank accesses.

Agent 2 measures the accepted Phase 9 belief head on `phase11_validation_bank_v1` and nothing else. It trains nothing, calibrates nothing, and moved no threshold, bin, baseline, bank or stratum. The sealed final-test bank was opened once, to re-hash its stored cases; no game, forward, score or truth touched it.

> **Readiness signal the reviewer should not miss.** On the validation bank, Gate A would not pass. `R_CE` is **0.9750** [0.9712, 0.9786] against Gate A's `<= 0.97` — the interval lies entirely above the threshold, so this is not sampling noise. The learned head *is* better than the count baseline (the CE-delta upper bound is -0.0463, comfortably negative, and Gates B, C and D all read as passing); it is simply not 3% better in cross-entropy. Validation values decide nothing and **nothing here was retuned in response** — Agent 7's sealed test on `phase11_test_bank_v1` is the verdict. But a reviewer authorizing Agent 3 should know that Phase 11 is, on current evidence, at real risk of a Gate A `FAIL`.

### 2.1 Verified identities

Every identity below was recomputed from live bytes at the start of the run.

```text
Agent 1 status                  PASS, 31/31 gates, zero problems
contract bundle                 ad16f921c602c1e1eb4975bee31fa6d1dff8dd4afdd09c332d9deaa94712192d
validation bank digest          bba6860549c05ebd59487d83d205e9d18b2109ab143d3816afbe793a13a04023
test bank digest                566ac35214ac04d5928af2f2975308a03bb78eb2a19e2ea05e6367f839eff404 (structural re-hash only)
Phase 9 checkpoint SHA-256      dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
Phase 9 model-state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 9 parameters              863,959
belief-head digest              a9df48a1adcd29b1a46c42ff1e605ede485119a36c247f1ae74f249f6d6f1dc7
global optimizer step           47,086 before, 47,086 after (delta 0)
P10-D selector config           6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668
Phase 7 library content         7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
Phase 10 closure commit         17188a5
```

### 2.2 The public/privileged boundary

The boundary is a type, not a convention. `Phase11BeliefRequest` carries exactly the five frozen fields and `from_payload` **raises** on anything else — an unknown field, a field whose name carries a frozen forbidden token, a forbidden key inside the public-state document. Nothing is dropped, because a dropped field is a leak that succeeded quietly.

The public-state document is built from an accepted `PublicView` plus the observation, so it has no access to a hidden rank at all. The suite proves this rather than asserting it: permuting the hidden army's true types leaves the document and its identity byte-identical.

True ranks arrive from somewhere else entirely. After a game ends and every learned and baseline vector already exists, a separate replay walks the public action history, rebuilds each recorded decision's document from scratch, and only then reads `record.true_type`. It writes one `int8` array to a separate `truth/` shard; a reader that never opens that directory has provably never seen a hidden rank.

```text
decisions re-derived            107,346 / 107,346
public-state identity mismatch  0
hidden-target alignment mismatch 0
remaining-count mismatches      0
legal-rank mask mismatches      0
unlabelled events               0
```

That is a 100% independent reconstruction of every recorded primitive — target set, square, moved flag, inventory, mask and the public-state identity itself — not a spot check.

### 2.3 The validation run

```text
games                     1,024 (512 cases x 2 colour games, exact)
observer decisions        107,346
prediction events         2,850,966
per stratum               128 games x 8 strata
per colour                red 512, blue 512
per setup source          neutral 512, p10d 512
backend                   CPU float32, 1 torch thread, greedy, single_request
wall clock                337.9s
```

Game outcomes are report-only and rank nothing: observer draw 16 (1.6%), loss 295 (28.8%), win 713 (69.6%).

### 2.4 Predictive metrics

Case-level percentile bootstrap, 10,000 replicates, 95%, both colour games pooled inside each case, one domain-separated PCG64 stream per metric token. Equal case weight, never equal event weight.

| metric | learned | `remaining_count_belief_v1` | delta (95% CI) |
| --- | --- | --- | --- |
| cross-entropy | 2.1050 | 2.1591 | -0.0541 [-0.0620, -0.0463] |
| top-1 accuracy | 0.2489 | 0.2146 | 0.0343 [0.0300, 0.0384] |
| Brier | 0.8403 | 0.8640 | -0.0237 [-0.0265, -0.0212] |
| true-rank probability | 0.1722 | 0.1345 | — |
| entropy (nats) | 2.0018 | 2.1676 | — |
| ECE (15 bins, pooled) | 0.0423 | 0.0020 | — |
| `R_CE` | 0.9750 [0.9712, 0.9786] | — | — |

2,850,966 events over 512 cases; 0 case(s) contributed no event. The CE floor fired on 0 event(s).

### 2.5 Per-stratum readings

| stratum | events | CE learned | CE baseline | `R_CE` | top-1 delta | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| `basic_rule` | 151,052 | 2.1071 | 2.1407 | 0.9843 | +0.0304 | 0.0369 |
| `information_miser` | 487,053 | 2.1301 | 2.1903 | 0.9725 | +0.0342 | 0.0419 |
| `miner_rush` | 590,591 | 2.1296 | 2.1726 | 0.9802 | +0.0323 | 0.0470 |
| `phase8_anchor` | 190,586 | 2.0798 | 2.1311 | 0.9759 | +0.0311 | 0.0413 |
| `phase9_selfplay` | 441,148 | 2.0234 | 2.1451 | 0.9433 | +0.0685 | 0.0371 |
| `scout_rush` | 576,331 | 2.2008 | 2.2174 | 0.9925 | +0.0132 | 0.0476 |
| `strategic_rule` | 208,004 | 2.1000 | 2.1432 | 0.9799 | +0.0268 | 0.0406 |
| `tactical_rule` | 206,201 | 2.0695 | 2.1327 | 0.9704 | +0.0380 | 0.0411 |

Full slices — observer colour, early/middle/late, moved/unmoved, per rank, per opponent setup source — are in `reports/phase_11_data/agent_02_predictive_metrics.json`; the stratum table is also `agent_02_stratum_metrics.csv`.

### 2.6 Validation readings of Gates A-D

**Diagnostic only.** The sealed test bank decides the gates; nothing here may move a threshold, and nothing here did.

| gate | requirement | validation reading | would pass |
| --- | --- | --- | --- |
| A | `R_CE <= 0.97` and CE-delta 95% upper `< 0` | 0.9750, upper -0.0463 | false |
| B | `Delta_top1 >= +0.03` and lower `> 0` | +0.03431, lower +0.03000 | true |
| C | ECE `<= 0.08`, no stratum `> 0.12`, Brier-delta upper `<= +0.01` | 0.0423, worst stratum 0.0476, upper -0.0212 | true |
| D | every stratum `R_CE <= 1.05` | worst 0.9925 | true |

### 2.7 Independent recomputation and negative controls

Three audit layers, each deliberately unlike what it checks.

```text
1. independent formulas   every one of 2,850,966 events; max deviation 8.882e-16
   (Brier by the algebraic identity, top-1 by an explicit scan, the
    softmax unshifted — different arithmetic, same limit)
2. pure-Python scalar     80,786 records over 32 cases from 64 games;
   max case-aggregate deviation 4.441e-16
3. the engine's counts    23,205 decisions, 638,105 hidden pieces checked against
   PublicView.unresolved_opponent_counts: 0 mismatches
```

All six required negative controls fire:

```text
hidden_truth_injected_into_request         fires
known_pieces_in_hidden_denominator         fires
permuted_probability_columns               fires
reversed_rank_mapping                      fires
wrong_remaining_inventory                  fires
wrong_true_rank_label                      fires
```

### 2.8 The two frozen baselines

`remaining_count_belief_v1` is mask-restricted count-proportional, with `c[r] = initial[r] - known[r]` over opponent pieces the observer legally knows alive or captured. Count conservation (`sum_r c[r]` = unresolved pieces) held at every one of the 23,205 audited decisions, the true rank never received zero mass, and every distribution matched an independent reconstruction exactly.

Edge-case coverage over the replayed games:

```text
capture                        21,958
moved_unknown                 129,892
near_endgame_exhaustion        13,834
public_scout_deduction          4,503
revealed_rank                  21,958
single_legal_rank                 126
```

`count_uniform_world_sampler_v1` produced 4,096 complete worlds over 512 distinct validation public states. Every one passed the frozen validation stack; every zero-tolerance counter is zero; every world re-derived exactly from its `(public-state identity, model label, sampler version, ordinal)` token; mean distinct worlds per state 7.63/8.

```text
dead_end_events                  0
hidden_input_accesses            0
immobility_violations            0
impossible_assignments           0
inventory_errors                 0
known_rank_violations            0
nonfinite_probability_rows       0
provenance_mismatches            0
public_knowledge_violations      0
```

Agent 2 built no learned sampler. The shared skeleton — piece order, the completion-feasibility guard, the inverse-CDF walk, the validation stack — is in place and tested, and `belief_sampler_v1` is Agent 3's to weight.

### 2.9 Preservation

```text
Phase 9 SHA / state / params    unchanged: true
belief-head identity            unchanged: true
C1 optimizer steps run          0
optimizer-counter delta         0
P10-D / utility / scaler        unchanged: true
Phase 7 library                 unchanged: true
Phase 8 anchor export           unchanged: true
```

### 2.10 Recorded readings

- **`optimizer_step_baseline_is_a_delta`** — the common contract's Gate H asks for 'C1 optimizer steps 0'. The accepted Phase 9 checkpoint already carries 47,086 historical steps, so the invariant Phase 11 can hold is a *delta* of exactly zero: no Phase 11 optimizer step, and the counter identical before and after. *(Gate H is read as a preservation delta, not an absolute.)*
- **`cases_without_events_are_excluded_from_the_case_mean`** — a case aggregate is the mean over the case's prediction events, which is undefined when a case has none — both its games ended before the observer ever acted. Such cases are excluded from the case mean and from resampling, and counted. On this run: 0 of 512. *(arithmetic, fixed before the run, identical for every metric.)*
- **`prediction_store_holds_logits_not_probabilities`** — the public shard stores the head's raw float32 logit rows. The frozen learned vector is the float64 softmax of exactly those rows, so logits are strictly more primitive and let the audit recompute the probabilities instead of trusting them. *(the recorded field set is unchanged; the storage is more primitive.)*
- **`public_action_history_is_stored_with_the_predictions`** — each public shard carries the game's absolute action ids. Every move is public to both players, so this adds no privileged information, and it lets the privileged pass and all three audits replay a game from its shard alone. *(no widening of the public boundary.)*
- **`edge_case_coverage_is_reported_not_asserted`** — the six frozen baseline edge cases are each constructed deterministically in the suite. Whether the replayed games also reach them is a fact about the bank and is reported: seen ['capture', 'moved_unknown', 'near_endgame_exhaustion', 'public_scout_deduction', 'revealed_rank', 'single_legal_rank'], unseen []. *(coverage is evidence, correctness is the gate.)*
- **`agent1_ledger_test_narrowed_to_the_frozen_rule`** — `tests/training/test_phase11_agent01_artifacts.py::test_the_ledger_proves_structural_only_access` asserted that *every* ledger entry carries agent=1 and structural_only=true. The frozen ledger rule says something narrower — every agent-harness bank access writes an entry, and every phase11_test_bank_v1 entry must be structural before Agent 7 — so Agent 2's scored validation-bank entry is the ledger working as designed. The test now asserts the contract: Agent 1's own entries are structural, and every test-bank entry is structural. Nothing about the seal was weakened. *(an over-strong Agent 1 test corrected to the frozen rule.)*
- **`eval_backend_frozen_to_cpu_float32_single_thread`** — MPS is not run-to-run bit-deterministic on this machine (the accepted Phase 8 finding), and Agent 4 must reproduce these decisions exactly, so the whole validation run is CPU float32 with one torch thread — the backend Agent 1 already froze for the runtime benchmark. *(the observer's greedy decision rule is unchanged.)*

### 2.11 Artifacts and completion gates

```text
reports/phase_11_data/agent_02_predictive_metrics.json
reports/phase_11_data/agent_02_stratum_metrics.csv
reports/phase_11_data/agent_02_baseline_audit.json
reports/phase_11_data/agent_02_acceptance.json
data/phase11/agent02/validation_predictions  (manifest 4246b156a023d847..., path is a diagnostic, never an identity)
data/phase11_prediction_root.txt  (tracked pointer)
```


The append-only ledger at `reports/phase_11_data/phase11_bank_access_ledger.jsonl` now carries 12 entries, 4 of them Agent 2's: two structural bank re-hashes, one 16-game smoke run taken while the harness was being built, and the 1,024-game acceptance run. Every one of the 5 `phase11_test_bank_v1` entries is structural with all four counters zero — the seal Agent 7 harvests.

Full suite: `.venv/bin/python -m pytest tests -q` — 5427 passed, 3 skipped in 357.52s (0:05:57)

| gate | value |
| --- | --- |
| `agent1_pass` | true |
| `all_required_prediction_events_recorded` | true |
| `baseline_negative_controls_fire` | true |
| `belief_head_unchanged` | true |
| `contracts_verified` | true |
| `count_uniform_world_baseline_complete` | true |
| `evaluator_negative_controls_fire` | true |
| `full_suite_green` | true |
| `independent_metric_recompute_pass` | true |
| `metrics_finite` | true |
| `no_belief_updates` | true |
| `no_test_prediction_access` | true |
| `no_test_truth_access` | true |
| `phase9_checkpoint_unchanged` | true |
| `prediction_schema_exact` | true |
| `public_privileged_boundary_pass` | true |
| `rank_order_exact` | true |
| `remaining_count_baseline_complete` | true |
| `test_bank_structural_only` | true |
| `validation_bank_verified` | true |
| `validation_color_balance_exact` | true |
| `validation_games_exact` | true |
| `validation_setup_source_balance_exact` | true |
| `validation_strata_exact` | true |

Agent 2 stops here and waits for reviewer acceptance. Agent 3 is authorized for the constrained sampler and its large audit over the validation public states; the test bank stays sealed with zero scored access, proven through the ledger.
