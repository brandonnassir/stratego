# Phase 5 Single-Agent Instructions

## Neural Model Contract and End-to-End Integration

**Assignment:** one implementation agent, working from the root of the real Stratego repository  
**Phase:** 5  
**Goal:** connect the frozen engine/observation/action system to a reusable PyTorch model/checkpoint/policy interface and prove the connection through the accepted Phase 4 evaluator  
**Training authorization:** no meaningful training; one controlled backward pass only

## 1. Read before editing

Read the repository's instructions and all of the following in full before writing code:

```text
stratego_project_docs/
reports/phase_2_implementation_report.md
reports/phase_2_metrics.json
reports/phase_3_implementation_report.md
reports/phase_3_data/
reports/phase_4_implementation_report.md
reports/phase_4_data/
```

Confirm these accepted contracts in the repository:

```text
rules:                   stratego_project_v1
reference engine:        phase2_1_reference_1.1.0
observation:             observation_v2_1_127ch
action encoding:         source_destination_10000_v1
policy interface:        policy_interface_v1
match identity:          match_spec_v1
setup bank:              evaluation_setup_bank_v1
pairing:                 color_swap_same_board
runner/statistics:       Phase 4 accepted versions
```

Before implementation, run the existing test suite and record the command, pass/fail/skip totals, runtime, current commit, operating system, Python, PyTorch, and device availability. If the suite fails before your changes, stop and report the pre-existing failure. Do not hide or repair unrelated failures.

## 2. Non-negotiable scope

You may add the neural integration layer, tests, scripts, Phase 5 report, and Phase 5 machine-readable results. You may make minimal integration re-exports where needed.

Do not:

- change rules, combat, legality, terminal precedence, observer knowledge, or state mutation;
- change the 127 channels, their semantics/order, or player-relative normalization;
- change the 10,000-action mapping;
- pass `GameState`, true hidden identities, opponent setup truth, replays, or belief targets to the model;
- alter Phase 4 pairing, setup bank, statistics, seeds, or baseline semantics;
- reuse the Phase 3 representative benchmark network as the model design;
- choose the Phase 6 architecture;
- start supervised learning, reinforcement learning, self-play training, optimizer tuning, or hyperparameter sweeps;
- silently reinterpret an incompatible checkpoint;
- substitute a legal action after a policy/model failure.

If the frozen contracts prevent implementation, stop and report the exact incompatibility with a minimal reproduction. Do not modify them to make Phase 5 easier.

## 3. Required software boundary

The permanent logical flow is:

```text
observer-safe PolicyInput
        ↓
[B, 127, 10, 10] float tensor
        ↓
pure normalized row-major tokenization
        ↓
[B, 100, 127]
        ↓
PyTorch model
        ├── policy logits [B, 10,000]
        ├── value logits  [B, 3]
        └── belief logits [B, 100, 12]
        ↓
authoritative engine legality mask
        ↓
greedy or seeded categorical choice
        ↓
PolicyResult
        ↓
independent engine validation
        ↓
ordinary Phase 4 match/evaluation
```

### Input

Canonical model input:

\[
X\in\mathbb{R}^{B\times127\times10\times10}.
\]

Convert it by a pure layout operation to:

\[
[B,127,10,10]\rightarrow[B,100,127].
\]

Token `i` is normalized row-major board square `i`. The transformation must not infer, reorder, append, normalize, or otherwise alter semantics.

### Policy output

Return `[B,10000]` logits. Index:

\[
a=100s+d
\]

means the engine move from source square `s` to destination square `d`. No remapping table is allowed. The model may score illegal indices; the adapter masks them with the authoritative engine product.

### Value output

Return `[B,3]` logits in the fixed order:

```text
WIN, DRAW, LOSS
```

These outcomes are from the acting/model player's perspective. After softmax:

\[
E[v]=P_W-P_L.
\]

Do not replace this with a scalar value head.

### Belief output

Return `[B,100,12]` logits over the 12 piece types. Define and test the loss mask for opponent-piece/square targets, including how own pieces, empty squares, and lakes are excluded. Belief targets are generated separately from privileged training state and never enter forward inference.

## 4. Required implementation

Prefer a dedicated package such as:

```text
stratego/model/
├── __init__.py
├── contract.py
├── tokenization.py
├── integration_model.py
├── checkpoint.py
├── losses.py
└── policy_adapter.py

tests/model/
├── test_contract.py
├── test_tokenization.py
├── test_policy_mapping.py
├── test_legality.py
├── test_checkpoint.py
├── test_hidden_information.py
├── test_value_belief.py
├── test_autograd.py
├── test_device_batch_equivalence.py
└── test_evaluation_integration.py
```

Adapt filenames to the repository only when necessary and document the final tree. Keep code readable for a beginner-to-intermediate Python user, with comments explaining non-obvious tensor shapes, perspective, masking, seeding, and compatibility checks. A small PyTorch module or policy adapter may use classes where the framework/interface requires them; keep surrounding utilities functional and simple.

### 4.1 Contract and tokenization

Define version constants and typed/validated output structures. Validate rank, dimensions, dtype family, finite outputs where required, and batch consistency. Freeze normalized row-major token ordering in tests using position-coded tensors so a transpose cannot accidentally pass.

### 4.2 Deliberately small fixture — `integration_model_v1`

Build a small, clearly labeled integration fixture. A reasonable default is:

- width 64;
- two Transformer encoder blocks;
- four attention heads;
- modest feed-forward width;
- shared encoder;
- policy, W/D/L value, and per-square belief heads.

This is a default, not a playing-strength choice. If repository constraints require a different small shape, explain it. Names and documentation must say that the fixture is not the final/production/Ataraxos model.

Initialize deterministically for tests. Disable dropout or use evaluation mode for deterministic inference.

### 4.3 Checkpoint-backed policy adapter

Implement a reusable neural policy that requests only:

```text
observation = true
legal action product = true
```

It must support:

1. deterministic greedy selection among legal actions;
2. seeded categorical selection using the Phase 4 per-decision seed contract.

Create the decision random stream once and reuse it. Do not recreate it for multiple random draws inside one decision. Define deterministic tie behavior. Reject empty legality, incompatible shapes, non-finite usable logits, and other invalid inputs loudly.

### 4.4 Checkpoint format

At minimum, save:

```text
checkpoint_format_version
model_architecture_id
model_contract_version
rules_version
observation_version
action_encoding_version
model_configuration
state_dict
training_iteration
training_step
creation_timestamp
optional_optimizer_state
optional_ema_state
optional_training_metrics
```

Validate all compatibility fields before model use. Reject missing required metadata, unknown/newer format versions, wrong rules/observation/action/model contracts, incompatible configurations, missing or unexpected weights, and corrupted/truncated files. Never silently load compatible-looking tensors under incompatible semantics.

On CPU, deterministic save/destroy/reload must reproduce policy, value, and belief logits bit-for-bit and select the same greedy action.

## 5. Required validation work

### 5.1 Exhaustive action audit

For each `action_id` from 0 through 9,999:

```text
decode -> source/destination -> encode -> same action_id
```

Also prove through the adapter that a crafted unique maximum at every legal tested index selects the same engine action. Ensure the test corpus covers every index for encode/decode and every legal move family for adapter selection.

### 5.2 Legality and numerical edge cases

Test at least:

- highest raw logit is illegal;
- all largest raw logits are illegal;
- exactly one legal action;
- tied legal maxima;
- extreme finite logits;
- float16 logits;
- `NaN`, positive infinity, and negative infinity;
- malformed and empty legal masks;
- the permanent Phase 3 Gumbel/non-finite sampler regression;
- independent engine rejection and inertness if an illegal action reaches it.

The adapter must never knowingly return an illegal action. Do not replace errors with random or first-legal moves.

### 5.3 Model-level hidden-information audit

Run at least **10,000 valid paired trials**. For each trial, create two privileged states with different unresolved opponent identities but identical acting-player information. Require:

- observations identical;
- legal actions identical;
- policy logits identical under fixed weights and deterministic CPU inference;
- value logits identical;
- belief logits identical;
- adapter diagnostics and greedy action identical;
- privileged belief targets different as the positive control;
- underlying permuted hidden types actually different.

Required mismatches/failures: zero. Report trial sources, game phases/plies, number of pieces permuted, skipped invalid/unchanged candidates, seeds, and full mismatch counts. The positive control must fire on every accepted trial.

### 5.4 Value and belief semantics

Use controlled logits to prove WIN/DRAW/LOSS ordering, probabilities summing to one, expected scalar `P_W-P_L`, and acting-player perspective for both colors. Validate belief shape, finiteness, target/mask shape, excluded squares, and separation of labels from inputs.

### 5.5 Autograd connectivity

Perform one controlled backward pass using placeholder policy, categorical value, and masked belief losses:

\[
L=L_{policy}+\lambda_vL_{value}+\lambda_bL_{belief}.
\]

Require finite total/component losses and finite gradients. Prove the shared encoder and all three heads receive the intended gradients. This is a connectivity smoke test, not a training experiment. Do not perform optimizer tuning or multi-step learning.

### 5.6 CPU, MPS, precision, and batch equivalence

Evaluate identical deterministic weights and inputs on:

```text
CPU float32
MPS float32
MPS float16
```

Record availability and skip status honestly if MPS cannot run on the actual target machine; a Phase 5 acceptance recommendation requires the intended M4 Pro/MPS validation to be completed eventually.

Predeclare tolerances in the script/report. A reasonable starting policy is `atol=rtol=1e-4` for float32 cross-device comparisons and `atol=rtol=5e-2` for float16, but tighten or adjust only with measured justification. Report maximum absolute/relative errors separately for policy logits, value probabilities, and belief logits.

Require finite outputs and exact legal-action agreement. Require exact greedy agreement on crafted-margin examples; report natural-corpus greedy agreement separately so near-ties are visible rather than hidden.

Evaluate the same position alone and embedded in batches of 8, 64, and 256. Compare the corresponding output row and selected action within predeclared dtype/device tolerances.

### 5.7 Minimal performance baseline

On MPS, measure inference at representative batches 1, 64, 256, and 1,024 after warmup. Report latency, positions/s, precision, synchronization method, memory/out-of-memory outcome, and whether tokenization/masking are inside or outside each timing. Do not tune the architecture against these results; Phase 6 owns the sweep.

### 5.8 Real Phase 4 gauntlet

Save a real `integration_model_v1` checkpoint, reload it through the reusable adapter, and run approximately 64 paired units against each accepted core baseline:

```text
random_legal@1.0.0
basic_heuristic@1.0.0
tactical_rule_based@1.0.0
strategic_rule_based@1.1.0
```

Use the accepted setup bank, `color_swap_same_board`, MatchSpec, seeds, runner, and paired reporting unchanged. Exercise both greedy and seeded categorical modes. Require:

- zero illegal actions;
- zero policy failures;
- correct color swap;
- checkpoint/model/version identity in results;
- exact replay/reproduction;
- deterministic greedy rerun;
- reproducible seeded-stochastic rerun.

Record win/draw/loss and effective win rate, but do not use playing strength as a pass gate.

## 6. Reports and machine-readable outputs

Append/create:

```text
reports/phase_5_implementation_report.md
```

Create:

```text
reports/phase_5_data/agent_01_phase5_acceptance.json
reports/phase_5_data/agent_01_hidden_information.json
reports/phase_5_data/agent_01_action_mapping.json
reports/phase_5_data/agent_01_checkpoint_compatibility.json
reports/phase_5_data/agent_01_numerical_batch_performance.json
reports/phase_5_data/agent_01_evaluation_gauntlet.csv
```

If the repository already has a naming convention that requires a small adjustment, keep the same logical separation and state the final paths.

### Report structure

Use these sections:

1. Status: PASS / CONDITIONAL PASS / FAIL.
2. Prerequisites and frozen-version verification.
3. Implementation summary and final file tree.
4. Contract definitions and deliberate non-decisions.
5. Tests added and all commands run.
6. Exhaustive action and legality results.
7. Hidden-information audit.
8. Checkpoint compatibility/save-load results.
9. Value, belief, and autograd results.
10. CPU/MPS, precision, batch, and performance results.
11. Phase 4 gauntlet and reproduction results.
12. Completion-gate table with evidence locations.
13. Known limitations/deviations.
14. Exact handoff recommendation for Phase 6.

Every headline number in Markdown must also exist in the machine-readable files. Include schema versions, timestamps, commit, platform, Python/PyTorch versions, seeds, commands, durations, and file digests where practical.

## 7. Twenty-two hard acceptance gates

Set each gate explicitly to true or false in `agent_01_phase5_acceptance.json`. Phase 5 may be recommended **PASS** only when all required gates are true.

1. `frozen_contracts_verified_unchanged` — rules, engine, observation, action, and Phase 4 semantics unchanged.
2. `preexisting_suite_green` — clean baseline recorded before edits.
3. `full_suite_green_after_changes` — all repository tests pass after implementation.
4. `input_shape_and_dtype_validated` — canonical `[B,127,10,10]` boundary enforced.
5. `tokenization_exact_row_major` — `[B,100,127]` conversion proven exact.
6. `policy_output_contract_validated` — `[B,10000]` logits and semantics correct.
7. `value_output_contract_validated` — `[B,3]`, WIN/DRAW/LOSS, acting-player semantics correct.
8. `belief_output_and_mask_validated` — `[B,100,12]` and loss-mask semantics correct.
9. `all_10000_actions_round_trip` — exhaustive encode/decode audit has zero mismatches.
10. `policy_index_matches_engine_action` — crafted adapter selections map to the same engine actions.
11. `legality_edge_cases_pass` — all masking, non-finite, tie, precision, and empty-set tests pass.
12. `engine_illegal_action_guard_preserved` — independent rejection is loud and inert.
13. `no_privileged_input_reachable` — object-graph and interface audit finds no privileged products.
14. `hidden_information_10000_zero_mismatch` — at least 10,000 valid model-level trials, zero mismatches, all positive controls valid.
15. `checkpoint_cpu_roundtrip_identity` — save/destroy/reload is bit-identical on CPU.
16. `checkpoint_incompatibilities_fail_loudly` — required negative/corruption/version tests pass.
17. `greedy_and_seeded_modes_reproducible` — both policy modes obey deterministic seed contracts.
18. `autograd_all_heads_connected_finite` — one multi-head backward pass is finite and connected.
19. `cpu_mps_float32_equivalence_pass` — cross-device comparison meets declared tolerances.
20. `mps_float16_finite_and_equivalent` — float16 inference is finite and meets declared tolerances.
21. `batch_equivalence_pass` — single versus batches 8/64/256 agree within declared tolerances.
22. `phase4_gauntlet_pass` — all four baselines complete with zero legality/policy/reproduction failures and correct metadata.

Performance measurements are required evidence but are not a strength/throughput threshold gate. If batch 1,024 is out of memory, report it honestly; smaller batches must still validate the pipeline.

## 8. Stop conditions

Stop implementation and report immediately if:

- a frozen engine/observation/action/evaluation change appears necessary;
- existing tests are already failing and the failure is not clearly environmental;
- privileged information is required to satisfy an interface;
- MPS produces persistent non-finite results or unexplained legality disagreement;
- a checkpoint can be loaded under incompatible semantic metadata;
- action mapping is ambiguous;
- the requested repository/report artifacts are absent, making evidence impossible to verify.

For an ordinary bug in new Phase 5 code, fix it, add a regression test, rerun affected and full suites, and document it. Do not stop merely because the work takes time.

## 9. Final response to the reviewing chat

Return:

- the Phase 5 status recommendation;
- the completion-gate table;
- exact test totals;
- the 10,000-trial hidden-information result;
- the 10,000-action audit result;
- checkpoint and device/batch results;
- gauntlet legality/reproduction result;
- paths to the Markdown report and every JSON/CSV artifact;
- any deviation, skipped environment-dependent gate, or frozen-contract concern.

Do not claim Phase 5 complete merely because the code runs. The reviewing chat must inspect the evidence and formally accept the phase before Phase 6 begins.
