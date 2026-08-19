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

## 3. Agent 3 — `belief_sampler_v1` and the Complete-World Audit

**Status: PASS** — 26/26 completion gates true, 251,262 learned worlds over 13,959 frozen validation public states (13,959 distinct), every zero-tolerance counter zero, zero optimizer steps, zero scored test-bank accesses.

Agent 3 implements the learned `belief_sampler_v1` exactly from the Agent 1 frozen mathematics — `weight = learned_probability * remaining_count`, the deterministic piece order, the inverse-CDF categorical walk, the counts-only zero-mass fallback and the completion-feasibility guard — and audits it at scale on the frozen validation public states. It runs no neural forward, plays no game, opens no truth shard, and reads no game outcome. The validation reading `R_CE = 0.9750` (a Gate A risk) was treated as diagnostic only: no belief weight, mask, baseline, sampler weighting, guard or threshold moved in response.

### 3.1 Verified identities

Every identity below was recomputed from live bytes before sampling.

```text
Agent 1 status                  PASS, 31/31 gates; Agent 2 PASS, 24/24 gates
contract bundle                 ad16f921c602c1e1eb4975bee31fa6d1dff8dd4afdd09c332d9deaa94712192d
sampler contract digest         a113d2e9588a6c4d7c2dcff954773e693ae876d19465904e4b277e86675afca9
validation bank digest          bba6860549c05ebd59487d83d205e9d18b2109ab143d3816afbe793a13a04023
test bank digest                566ac35214ac04d5928af2f2975308a03bb78eb2a19e2ea05e6367f839eff404 (structural re-hash only)
prediction-store manifest       4246b156a023d8475448e5e7a6f276ad6938dda66aaec0bf655c436e391634d7
Phase 9 checkpoint SHA-256      dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
Phase 9 model-state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 9 parameters              863,959
belief-head digest              a9df48a1adcd29b1a46c42ff1e605ede485119a36c247f1ae74f249f6d6f1dc7
P10-D config SHA-256            6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668
Phase 7 library content         7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
```

### 3.2 The learned sampler and the shared skeleton

`stratego/evaluation/phase11_sampler.py` implements the twelve frozen steps on the accepted skeleton primitives (`feasible_ranks`, `inverse_cdf_choice`, `validate_world`), so the learned sampler and the accepted `count_uniform_world_sampler_v1` differ in exactly one place: step 7's weight. The request boundary is a frozen dataclass with the four allowed fields and nothing else; `from_payload` raises on any unknown field, and the audit probed every named rejected input (12 probes, all refused).

```text
stratego/evaluation/phase11_sampler.py           a0119f0126a1100c3fd74a20a703ea47c183d43e7fb1b6822aa15c7e34b921e8
stratego/evaluation/phase11_sampler_audit.py     26fc32d8428942a9b09f1cdfda5f7e5455bd34e69c19a2a02027eaacc866b87a
```

**The completion-feasibility guard reads public constraints only.** Its three inputs — `movable_remaining` (public inventory summed over movable ranks), `moved_unresolved_remaining` (public `has_moved` flags over the not-yet-assigned pieces), and the current piece's public mask — are all derived from the public-state document; the request type has no field hidden truth could arrive in, the injection controls were rejected structurally, and the independent path recomputed the guard from the raw document on 763,863 steps with zero disagreements (4,558 of those steps visibly pruned a movable rank, and every pruned walk still completed).

### 3.3 The large audit

```text
states sampled                  13,959 (13,959 distinct identities)
learned worlds                  251,262 (floor 250,000)
worlds validated                251,262 (100%)
baseline worlds                 55,836 (count_uniform_world_sampler_v1, same states)
independent second-path worlds  25,127 (floor 25,000)
strata covered                  8/8; colours ['blue', 'red']
progress buckets                {'early': 5296, 'late': 4169, 'middle': 4494}
moved/unmoved uncertainty       12,221 / 13,959 states
unresolved pieces               mean 30.404, max 40
wall clock                      275s (1115.6 worlds/s)
```

Zero-tolerance counters, learned sampler (all must be and are zero):

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

Store integrity: 1,024 public shards re-hashed against the Agent 2 manifest (0 mismatches); every selected decision's rebuilt document matched its stored identity, observation digest, hidden-slot set, masks and counts exactly (0/0/0 identity/mask/count mismatches). The baseline sampler produced 55,836 valid worlds on the same states with all counters zero; no strength comparison was run.

### 3.4 Independent audit, determinism, collisions, controls

The second implementation path (`stratego/evaluation/phase11_sampler_audit.py`) imports no Phase 11 module: it rebuilds the inventory, masks, multiset, public facts and the raw-`blake2b` seed derivation from the engine authority and the published contract text, and re-runs every audited walk with scalar arithmetic. 25,127 worlds re-derived exactly, 0 disagreements, 0 float knife-edge events across 763,863 recomputed steps.

```text
deterministic repeats           10,062 worlds re-sampled bit-identically (0 mismatches)
call-order reversal             559 states re-sampled in reverse ordinal order (0 mismatches)
seed collision audit            19,152,614 seeds (19,152,614 distinct), no collisions: True
  world_sample                  307,098 derived, 307,098 distinct
  world_order                   9,337,152 derived, 9,337,152 distinct
  world_categorical             9,337,152 derived, 9,337,152 distinct
```

The collision audit ran `stream_collision_audit` over every `world_sample`, `world_order` and `world_categorical` seed the audit actually derived — exhaustively, learned and baseline tokens alike — combined with the complete re-derived Agent 1 enumerable universe (bank, soak, safety, repro, benchmark, bootstrap), so the frozen downstream obligation is discharged against the whole relevant seed space, not just the new streams.

Negative controls (each must fire and did):

```text
remove_one_remaining_rank        fired=True
bomb_or_flag_on_moved_piece      fired=True
duplicate_marshal_count          fired=True
alter_public_known_rank          fired=True
mutate_sample_seed               fired=True
inject_true_hidden_rank          fired=True
corrupt_provenance               fired=True
```

### 3.5 Report-only diagnostics

Zero-mass fallback: 0 steps in 0 of 251,262 worlds (rate 0.0). Per-state distinct-world counts, empirical-vs-learned marginal L1 agreement and learned entropy are in `agent_03_sampler_diagnostics.csv` (13,959 rows). No diversity threshold is frozen, so these rank nothing.

### 3.6 Preservation and the seal

```text
Phase 9 checkpoint unchanged    True
belief head unchanged           True
optimizer step delta            0 (steps run: 0)
P10-D / anchor / Phase 7        True / True / True
prediction store unchanged      True
bank artifacts unchanged        True
test bank                       structural-only, 0 scored / 0 truth / 0 outcome / 0 inference accesses
```

### 3.7 Completion gates

| # | Gate | Result |
|---|------|--------|
| 1 | `agents1_2_pass` | true |
| 2 | `all_8_strata_covered` | true |
| 3 | `all_zero_tolerance_counters_zero` | true |
| 4 | `baseline_world_sampler_valid` | true |
| 5 | `both_colors_covered` | true |
| 6 | `categorical_draw_seeded` | true |
| 7 | `complete_world_validation_exact` | true |
| 8 | `deterministic_repeat_pass` | true |
| 9 | `exact_inventory_enforced` | true |
| 10 | `feasibility_guard_public_inputs_only` | true |
| 11 | `full_suite_green` | true |
| 12 | `independent_audit_pass` | true |
| 13 | `known_ranks_locked` | true |
| 14 | `negative_controls_fire` | true |
| 15 | `no_belief_updates` | true |
| 16 | `no_test_prediction_access` | true |
| 17 | `piece_order_seeded` | true |
| 18 | `public_masks_enforced` | true |
| 19 | `sampler_contract_verified` | true |
| 20 | `sampler_request_boundary_exact` | true |
| 21 | `sampler_worlds_ge_250k` | true |
| 22 | `thousands_distinct_states` | true |
| 23 | `true_hidden_inputs_rejected` | true |
| 24 | `upstream_artifacts_unchanged` | true |
| 25 | `world_stream_collisions_zero` | true |
| 26 | `zero_mass_fallback_exact` | true |

Suite: `5497 passed, 3 skipped in 313.78s (0:05:13)`.

### 3.8 Recorded readings and handoff to Agent 4

- **gate_a_risk_acknowledged_nothing_retuned** — Agent 2's validation reading R_CE = 0.9750 would fail Gate A's <= 0.97 on the sealed test if it repeated. Agent 3 treats this as diagnostic only: the belief model, masks, baseline, sampler weighting (learned_probability * remaining_count), feasibility guard and every Phase 11 threshold are byte-identical to the Agent 1 freeze. Nothing was retuned in response.
- **audit_schedule_frozen_before_sampling** — the audit samples the 16 evenly spaced eligible decisions of every validation game (floor(k*E/n), the accepted benchmark/soak spacing) and 18 learned worlds per state under W = max(16, ceil(250,000 / states)); realized 13959 states x 18 = 251,262 worlds. The rule was frozen in the contract artifact before any world existed and satisfies the contract floors; it moves no frozen threshold.
- **sampler_audit_replays_are_structural** — the audit replays recorded public action histories through the engine to rebuild frozen public-state documents; no new game is played, no neural forward runs, no prediction is scored, no truth shard is opened, and the manifest's outcome fields (observer_result, terminal_reason) are not read. The ledger entry is therefore structural_only=true with all four counters zero.
- **world_sample_root_seed_derived_for_the_collision_audit** — the frozen walk consumes the world_order and world_categorical child streams; the world_sample root seed of every materialized token is additionally derived and collision-checked, because the contract's downstream obligation names all three streams.
- **independent_float_path_and_knife_edges** — the independent path re-runs the categorical walk with scalar arithmetic (math.fsum totals) against the primary's NumPy sums. The two can only disagree when a draw lands within a few ulps of a bin boundary; such knife-edge steps are counted and the audit observed 0 across 763,863 recomputed steps, with zero assignment disagreements.
- **baseline_verification_scope** — count_uniform_world_sampler_v1 was verified on the same 13,959 states (55,836 worlds, all counters zero). No strength comparison was run, as the contract requires.

Agent 4 receives the immutable sampler identity (`stratego/evaluation/phase11_sampler.py`, SHA-256 `a0119f0126a1100c...`), the provenance schema, the sample-token rules (production ordinals 0..63), the validation public-state list (the diagnostics CSV), the audit evidence, and the zero-mass fallback behaviour. Agent 4 must not change the sampler mathematics.

**Agent 3 stops here.** Ending revision: uncommitted working tree over `2c12a5c`; per the commit discipline, the commit happens only after reviewing-chat acceptance.

## 4. Agent 4 — Information Safety, Reproducibility, and Runtime

**Status: PASS** — 34/34 completion gates true, 50,000 hidden-truth permutation trials with every information-safety counter zero, all 8 topology/restart legs byte-identical over 2,048 frozen requests (131,072 complete worlds per leg), and p95(forward + 64 worlds) = 48.5 ms against the 500 ms ceiling.

Agent 4 proves the three properties Phase 12 needs from the belief system: it cannot see hidden truth, it reproduces exactly under every required topology and restart, and it is fast enough for search. It trains nothing, calibrates nothing, redesigns no sampler rule, touches no P10-D artifact and scores no test-bank prediction. The validation `R_CE = 0.9750` Gate A risk stays diagnostic: every threshold this agent tested against — 50,000 trials, eight legs, 500 ms — is Agent 1's, unchanged.

### 4.1 Verified identities

Every identity below was recomputed from live bytes before any measurement.

```text
Agent 1 status                  PASS; Agent 2 PASS; Agent 3 PASS
contract bundle                 ad16f921c602c1e1eb4975bee31fa6d1dff8dd4afdd09c332d9deaa94712192d
information-safety contract     1b8160d544b5ee71eb1b03be025a868e7298ace1d61c486524e631ba68faab4d
sampler contract digest         a113d2e9588a6c4d7c2dcff954773e693ae876d19465904e4b277e86675afca9
validation bank digest          bba6860549c05ebd59487d83d205e9d18b2109ab143d3816afbe793a13a04023
test bank digest                566ac35214ac04d5928af2f2975308a03bb78eb2a19e2ea05e6367f839eff404 (structural re-hash only)
prediction-store manifest       4246b156a023d8475448e5e7a6f276ad6938dda66aaec0bf655c436e391634d7
Phase 9 checkpoint SHA-256      dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
Phase 9 model-state digest      f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
Phase 9 parameters              863,959
belief-head digest              a9df48a1adcd29b1a46c42ff1e605ede485119a36c247f1ae74f249f6d6f1dc7
stratego/evaluation/phase11_sampler.py  a0119f0126a1100c3fd74a20a703ea47c183d43e7fb1b6822aa15c7e34b921e8
stratego/evaluation/phase11_sampler_audit.py  26fc32d8428942a9b09f1cdfda5f7e5455bd34e69c19a2a02027eaacc866b87a
P10-D config SHA-256            6e227815bc3cb44f19cdeee55d00ec0ae75726fb411ee9131660aa712bb86668
Phase 7 library content         7b8a66601ce5874a95e81233e4924db186839402093936baafc7776e61b02777
```

### 4.2 The frozen sets

Both sets were materialised from Agent 1's hash-order rules over the Agent 2 prediction store — no seed stream, no clock, no replay — and written before any trial, leg or measurement existed.

```text
topology request set            2,048 requests, 2,048 distinct public states
  digest                        83e58f00984b427d9ecbb09d40b343297dce0717898e3599fbbfcf26667bcd84
  per stratum                   256 x 8
  observer colour               {'blue': 969, 'red': 1079}
  progress bucket               {'early': 431, 'late': 1046, 'middle': 571}
  unresolved pieces             1-40
benchmark state set             480 states over 48 cells, 0 short
  digest                        1ec642e8c2b509848d0db1565b052bc7e892c71cc3ab8e8764de6eb7b2869a48
  unresolved pieces             1-40 (40 distinct counts)
safety candidate pool           107,190 candidates, 107,021 admitting an altered legal truth
  digest                        5512bc56c18e6aac2cafe641dd064a537b8b6659cbc5fb16135cf31de12e5552
```

### 4.3 Part A — the hidden-truth permutation attack

Each trial takes a validation position, permutes the true ranks of the unresolved opponent pieces into a different but publicly indistinguishable truth, and re-runs the production belief path and the frozen sampler on both. The permutation preserves the remaining inventory by construction and never puts a Flag or a Bomb on a publicly moved piece.

```text
trials                          50,000 (floor 50,000)
belief forwards                 100,000
fixed-seed worlds               100,000
instrumented public rebuilds    100,000
distinct positions              39,895 over 939 games
changed hidden ranks per trial  mean 20.3491, max 40
strata                          {'basic_rule': 2822, 'information_miser': 7969, 'miner_rush': 10965, 'phase8_anchor': 3595, 'phase9_selfplay': 6990, 'scout_rush': 10004, 'strategic_rule': 3824, 'tactical_rule': 3831}
observer colour                 {'blue': 25120, 'red': 24880}
progress bucket                 {'early': 7939, 'late': 29599, 'middle': 12462}
wall clock                      762s
```

Gate F zero-tolerance counters (all must be and are zero):

```text
belief_output_differences               0
fixed_seed_sample_differences           0
forbidden_hidden_input_accesses         0
injection_acceptances                   0
```

The six contract checks are recorded separately, so a difference could not hide inside an aggregate:

```text
belief_logit_probability_differences    0
illegal_alternative_truths              0
instrumented_document_mismatches        0
inventory_changes                       0
legal_action_mask_differences           0
observation_differences                 0
public_document_differences             0
public_mask_differences                 0
sampled_world_differences               0
sampler_provenance_differences          0
sampler_request_differences             0
unchanged_alternative_truths            0
```

**Injection controls.** 288 probes across 8 positions pushed every named private field — true rank, private piece table, opponent setup truth, hidden start rank, winner/result/reward, future action/search result and storage path — at *both* request boundaries, including two nested smuggles that hide a private field inside the frozen public document. Every probe was refused structurally; `injection_acceptances = 0`.

**The hidden-rank access counter is instrumented, not asserted.** Each trial rebuilds both positions' public products a second time with the unresolved opponent pieces replaced by records whose `true_type` is a counting property, so any read of a hidden rank while building the `PublicView`, the 127-channel observation or the frozen document is tallied. Across 100,000 instrumented rebuilds the count is 0, and every instrumented document matched its plain counterpart byte for byte.

### 4.4 Part B — topology and restart reproducibility

One definition of a request serves every leg and the benchmark (`stratego.evaluation.phase11_repro.execute_request`): one belief forward plus complete worlds for sample ordinals 0..63, summarised by a SHA-256 over the raw bytes of the logits, the float64 probabilities, the public legal-rank masks, all 64 worlds and every provenance field. Each leg replays every request from the initial setup, so nothing survives between requests.

```text
leg                              workers  requests   seconds  rollup digest
workers_1                              1      2048      91.5  267c20efb9fe3456
workers_4                              4      2048      25.2  267c20efb9fe3456
workers_12                            12      2048      11.2  267c20efb9fe3456
forward_order                          1      2048      89.7  267c20efb9fe3456
reverse_order                          1      2048      90.2  267c20efb9fe3456
round_robin_sharded                    5      2048      19.7  267c20efb9fe3456
fresh_process                          1      2048      91.4  267c20efb9fe3456
kill_resume_set_subtraction            1      2048      92.1  267c20efb9fe3456
```

All 8 legs produced exactly one rollup digest (`267c20efb9fe34569ec23df29eee76156ec03cea7fd2230d088b66920614febb`), with 0 request mismatches against the reference leg in every comparison. The restart leg sent a real `SIGKILL` after 192 fsynced requests and resumed with exactly the 1856 ordinals the store did not hold; 0 requests were recomputed on both sides of the kill.

**No mutable RNG cursor exists on the production path.** A literal scan over the live source of 6 derivation modules for 19 markers — module-level `random`/`numpy.random`/`torch` draws, wall clock, process id, `os.urandom`, `uuid4` — returned no findings, and the `mutable_global_rng` sensitivity control shows what a cursor-driven order would have done to the leg comparison.

### 4.5 Part C — the runtime benchmark

Backend and device were frozen by Agent 1 before any measurement existed (cpu / float32 / 1 torch thread, single process, single request at a time) and did not move after results. 32 global warmups and one discarded warmup per state precede the measurements; the timer is `time.perf_counter_ns` around the complete request.

```text
configuration                median      p90      p95      p99      max   forward  sampling
forward_only                    1.9      2.1      2.1      2.2      2.3      1.52      0.11
forward_plus_16_worlds         11.9     13.2     13.4     13.8     14.1      1.29     10.38
forward_plus_32_worlds         22.2     24.7     25.1     25.6     26.3      1.35     20.58
forward_plus_64_worlds         42.7     47.8     48.5     49.7     53.1      1.42     41.01
(milliseconds; forward and sampling columns are medians of the components)
```

**Gate G quantity: p95(forward + 64 worlds) = 48.51 ms <= 500 ms**, a 10.31x headroom, at a peak RSS of 255.7 MiB over 480 states and 1,920 measured requests. Every recorded metric is finite.

### 4.6 Part D — sensitivity controls

Each control sabotages one thing the evidence depends on; each must fire, and each did.

```text
belief_probability_perturbed            fired
mutable_global_rng                      fired
private_truth_read                      fired
provenance_corrupted                    fired
sample_seed_changed                     fired
```

### 4.7 Materialized random-stream identities

Every Phase 11 draw is a `blake2b` of a logical identity, so two different identities sharing a seed would silently couple two independent draws. Agent 3 proved injectivity over its own world streams and Agent 1's enumerable universe; Agent 4 materializes identities neither covered — sample ordinals up to 63 on states Agent 3 never sampled, and `safety_trial` draws beyond ordinal 0, which is all Agent 1 enumerates.

Intentional reuse is deduplicated by logical identity before any seed is compared: the original and permuted sides of a safety trial share one sampler identity **by design**, the eight legs reissue identical request and sample identities **by design**, and Agent 1's draw-0 safety entries are the first draw of the same trial streams the attack consumed. What remains is one entry per distinct identity.

```text
sample tokens                            count
  Agent 3 (reconstructed)              307,098
  Agent 4 safety attack                 49,820
  Agent 4 topology legs                131,072
  Agent 4 runtime benchmark             30,720
  Agent 4 controls (subset)                  8
  Agent 4 distinct                     209,248
  new to Agent 4                       199,941
  shared with Agent 3                    9,307
  combined distinct                    507,039
```

Per-domain identity counts of the combined universe:

```text
domain                               identities  distinct seeds  internal dup
bank_match                                5,120           5,120             0
bank_observer_setup                       5,120           5,120             0
bank_opponent_setup                       5,120           5,120             0
benchmark:state_selection                   480             480             0
bootstrap                                   252             252             0
repro_schedule:replay                     2,048           2,048             0
safety_trial:sample_check                50,000          50,000             0
safety_trial:state_selection             50,063          50,063             0
safety_trial:truth_permutation          362,655         362,655             0
soak_match                                1,024           1,024             0
soak_setup                                2,048           2,048             0
world_categorical                    14,621,713      14,621,713             0
world_order                          14,621,713      14,621,713             0
world_sample                            507,039         507,039             0

combined                             30,234,395      30,234,395             0
```

**30,234,395 distinct logical identities map to 30,234,395 distinct seeds — 0 accidental collisions**, of which 11,231,781 are Agent 4's own new identities. The `repro_schedule` and `benchmark` domains are materially uninstantiated (both frozen selection rules are hash-order rules that consume no randomness), so they contribute only Agent 1's enumerable entries.

Two things make the combination trustworthy rather than merely large. The Agent 3 universe is *reconstructed* from its diagnostics and the recorded store, and the reconstruction reproduces its recorded stream counts exactly (world_categorical 9,337,152, world_order 9,337,152, world_sample 307,098). And the bulk enumeration calls `derive_phase11_seed` directly to avoid tens of millions of token re-parses, so 15,666 of those derivations were re-run through the accepted public helpers (`world_sample_seed`, `world_order_key`, `world_categorical_uniform`, `safety_trial_seed`) with 0 mismatches.

The attack's own stream consumption was recomputed in a fresh process from the frozen pool alone and reproduced the recorded run's permutation-method split exactly ({'shuffle': 44148, 'transposition': 5852}), which is an independent determinism check on Part A.

### 4.8 Preservation and the seal

```text
Phase 9 checkpoint unchanged    True
belief head unchanged           True
sampler identity unchanged      True
optimizer steps                 47086 -> 47086 (delta 0)
P10-D unchanged                 True
Phase 7 library unchanged       True
prediction store unchanged      True
test-bank entries               11, structural-only True
test-bank counters              forwards 0, scored 0, truth 0, outcomes 0
```

### 4.9 Completion gates

```text
agent4_materialized_stream_collisions_zeroTrue
agents1_3_pass                          True
all_topology_legs_exact                 True
belief_head_unchanged                   True
belief_output_changes_zero              True
benchmark_config_frozen                 True
benchmark_states_representative         True
fixed_seed_sample_changes_zero          True
forbidden_hidden_access_zero            True
forward_reverse_exact                   True
fresh_process_exact                     True
full_suite_green                        True
hidden_truth_trials_ge_50k              True
injection_controls_rejected             True
mutable_rng_absent                      True
negative_controls_fire                  True
no_belief_updates                       True
no_test_prediction_access               True
one_distinct_rollup_digest              True
p95_64_worlds_le_500ms                  True
p95_64_worlds_recorded                  True
phase9_checkpoint_unchanged             True
recorded_logits_reproduce_exactly       True
restart_resume_exact                    True
round_robin_sharded_exact               True
runtime_metrics_finite                  True
safety_detail_counters_zero             True
sampler_identity_unchanged              True
stream_universe_reconstruction_faithful True
topology_request_set_frozen             True
upstream_artifacts_unchanged            True
worker_12_exact                         True
worker_1_exact                          True
worker_4_exact                          True
```

Suite: `5621 passed, 3 skipped in 315.50s (0:05:15)`

### 4.10 Recorded readings and handoff to Agent 5

- **gate_a_risk_acknowledged_nothing_retuned** — Agent 2's validation reading R_CE = 0.9750 would fail Gate A's <= 0.97 on the sealed test if it repeated. Agent 4 treats it as diagnostic only: the belief model, the masks, the baseline, the sampler weighting, the feasibility guard and every Phase 11 threshold are byte-identical to the Agent 1 freeze, and this agent's own thresholds (50,000 trials, eight legs, 500 ms) are Agent 1's. *Impact:* none on any frozen quantity; recorded so the reviewer sees the risk was known and not acted on
- **permutation_attack_reads_truth_on_its_construction_path** — the attack must build an alternative hidden truth, so it reads the validation position's true ranks on its own privileged construction path — the contract's `permutation` rule requires exactly this. Those ranks never enter a belief request, a sampler request or any derivation: both request types have no field they could arrive in, and the instrumented counter proves the public products were built without a single hidden-rank read. No truth shard was opened and no game outcome was read. *Impact:* validation-bank access accounting only; the test bank is untouched
- **admits_alternative_is_the_constructive_predicate** — a candidate state is usable when a valid transposition of two unresolved pieces exists — different ranks, and neither piece left publicly-moved-and-immovable. A transposition *is* an alternative truth, so the predicate is sufficient by construction, and it is the exact rule the frozen no-alternative walk uses. On this pool 169 of 107190 candidates were skipped by it; no trial was dropped. *Impact:* which states a trial may land on; never which comparison it runs
- **belief_request_boundary_hardened_to_the_frozen_document_schema** — the injection controls found one gap: `Phase11BeliefRequest` scanned only the *top-level* document keys for forbidden tokens, so a private field nested inside a piece entry (`pieces[0]['true_rank_index']`) was accepted, while the sampler boundary refused the same payload. The frozen rule is 'requests carrying private fields must be rejected structurally', and such a request carries one, so the belief request now applies the same exact-schema refusal the sampler already did, over the document, its pieces and its recent moves. This adds a refusal and touches no arithmetic: the recorded-logit agreement pass re-ran the live forward on all 2048 frozen requests and reproduced Agent 2's stored float32 logits byte for byte on 53868 rows, with 0 mismatches. Every document Phase 11 builds already satisfies the schema, because the accepted builder raises on drift itself. *Impact:* a refusal added to an accepted request type; no metric, threshold, weight or recorded output moved
- **repro_schedule_and_benchmark_domains_are_not_instantiated** — Agent 1 froze a `repro_schedule` and a `benchmark` stream so that any schedule step needing a draw would have a domain-separated source instead of an invented one. Neither frozen selection rule needs one: the 2,048-request set is the distinct validation public states ordered by identity, and the 480-state benchmark orders each cell by unresolved count then identity. The harness therefore calls `repro_schedule_seed` and `benchmark_seed` nowhere, and the stream audit records both domains as materially uninstantiated while still carrying Agent 1's enumerable entries into the combined injectivity check. *Impact:* the two domains contribute their frozen enumerable entries only; no draw was invented
- **measured_request_includes_public_product_construction** — the benchmark timer wraps the whole request as Phase 12 will issue it: build the public view, the 127-channel observation and the frozen document, run the forward, then sample the worlds. The engine replay that puts the harness at the position is excluded — a searcher already holds the position — and the document, forward and sampling components are recorded separately so the split is visible rather than asserted. *Impact:* makes the measured quantity conservative; the ceiling is unchanged
- **forward_order_is_the_reference_leg** — the eight legs are compared pairwise against `forward_order`, the only leg that runs inside the harness process itself, and it runs there after that process has already executed the 2048-request recorded-logit agreement pass and driven the three worker legs — so it is the leg most exposed to 'previous calls', and comparing every other leg to it is strictly harder than comparing them to a fresh process. All 8 legs produced one rollup digest: 267c20efb9fe34569ec23df29eee76156ec03cea7fd2230d088b66920614febb. *Impact:* comparison bookkeeping; every leg is compared to every other through it
- **kill_resume_uses_a_real_sigkill** — the restart leg starts a real subprocess, waits until it has fsynced 192 committed requests, sends SIGKILL, then resumes with exactly the ordinals the store does not hold. 0 requests were recomputed on both sides of the kill, and the union is the complete frozen set. *Impact:* none; it is the restart evidence itself

Agent 5 receives the immutable evaluator identity (`phase11_belief_evaluator_v1`, belief head `a9df48a1adcd29b1...`), the immutable sampler identity (`belief_sampler_v1`, `a0119f0126a1100c...`), the safety and topology evidence, and the measured runtime configuration (cpu / float32 / 1 thread, p95 48.51 ms). Agent 5 integrates and freezes; it may not retrain, recalibrate, redesign the sampler or open the sealed test bank.

