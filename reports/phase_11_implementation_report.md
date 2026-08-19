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

