# Phase 5 Implementation Report

## Neural Model Contract and End-to-End Integration

Frozen reference: `phase2_1_reference_1.1.0`
Rules: `stratego_project_v1`
Observation: `observation_v2_1_127ch`
Action encoding: `source_destination_10000_v1`
Policy interface: `policy_interface_v1`
Model contract: `model_contract_v1` (new)
Model architecture: `integration_model_v1` (integration fixture, **not** the final model)

Agent: single implementation agent (`agent_01`)
Commit under test: `1d8e7cb4473e4bb037aff6a70d162effe3457c68`
Regenerate everything: `python scripts/run_phase5.py` (736 s end to end)

---

## 1. Status

**PASS** — all 22 hard acceptance gates are true.

This is a recommendation, not an acceptance. The reviewing chat must inspect the
evidence below and formally accept Phase 5 before Phase 6 begins. Phase 5 claims
only that the frozen engine/observation/action system drives a real PyTorch
model through a real checkpoint into the real Phase 4 evaluator, correctly and
reproducibly. **It makes no claim about playing strength**, and the fixture
network is untrained and weak by construction.

| Headline | Result |
|---|---|
| Pre-existing suite, before any edit | 1,963 passed, 0 failed, 2 skipped, 64.0 s |
| Full suite after implementation | **2,155 passed, 0 failed, 2 skipped**, 104.1 s |
| New tests added | 192, all passing |
| Exhaustive action audit | 10,000 / 10,000 round-trip, **0 mismatches** |
| Adapter selection audit | 264 crafted selections, **0 mismatches** |
| Model-level hidden-information audit | **10,000 trials, 0 mismatches, 0 positive-control failures** |
| Checkpoint CPU round trip | bit-identical on all three heads |
| Checkpoint incompatibilities | 19 negative cases + 5 corrupted files, **24 / 24 rejected** |
| CPU ↔ MPS float32 | within declared tolerance, max abs error 7.15e-07 |
| MPS float16 | finite, within declared tolerance, max abs error 1.75e-03 |
| Batch equivalence 1 vs 8 / 64 / 256 | exact on policy and belief, 1.19e-07 on value |
| Phase 4 gauntlet | 1,024 matches, 603,874 plies, **0 illegal actions, 0 policy failures** |
| Reproduction | greedy and seeded-stochastic reruns both digest-identical |

---

## 2. Prerequisites and frozen-version verification

### 2.1 Clean baseline, recorded before the first Phase 5 file existed

```text
command   python -m pytest -q
commit    1d8e7cb4473e4bb037aff6a70d162effe3457c68
result    1963 passed, 2 skipped in 64.01s
```

The two skips are pre-existing and unrelated to Phase 5 — both are
`tests/evaluation/test_baseline_information_safety.py:219`, skipped because
`random_legal` and `stress_chaos` do not expose a per-move score vector. They
were **not** repaired or hidden; they appear unchanged in the post-implementation
run. The baseline is recorded in `agent_01_phase5_acceptance.json` under
`preexisting_suite` so gate 2 cites a measurement rather than an assertion.

### 2.2 Environment

| Item | Value |
|---|---|
| Platform | macOS 26.5.2, arm64 (Apple M4 Pro) |
| Python | 3.13.2 |
| PyTorch | 2.13.0 |
| NumPy | 2.5.1 |
| MPS built / available | true / **true** |
| CUDA | not available |
| CPU threads | 10 |

The intended M4 Pro / MPS validation ran on the actual target machine. **No
device-dependent gate was skipped.**

### 2.3 Frozen contracts, verified unchanged

| Contract | Required | Found |
|---|---|---|
| rules | `stratego_project_v1` | `stratego_project_v1` |
| reference engine | `phase2_1_reference_1.1.0` | `phase2_1_reference_1.1.0` |
| observation | `observation_v2_1_127ch` | `observation_v2_1_127ch` |
| observation channels | 127 | 127 |
| action encoding | `source_destination_10000_v1` | `source_destination_10000_v1` |
| action space | 10,000 | 10,000 |
| policy interface | `policy_interface_v1` | `policy_interface_v1` |
| setup bank | `evaluation_setup_bank_v1` | `evaluation_setup_bank_v1` |
| pairing | `color_swap_same_board` | `color_swap_same_board` |
| suite | Phase 4 accepted | `phase4_evaluation_suite_v1` |

No file under `stratego/engine/` or `stratego/evaluation/` was modified. Rules,
combat, legality, terminal precedence, observer knowledge, state mutation, the
127 channels, the 10,000-action mapping, Phase 4 pairing, the setup bank,
statistics, seeds and baseline semantics are all untouched. The Phase 3
representative benchmark network was **not** reused as the model design.

---

## 3. Implementation summary and final file tree

`stratego/engine/`, `stratego/evaluation/` and the existing `stratego/training/`
modules were **not modified**. Phase 5 is additive: a new `stratego/model/`
package, one new privileged-target module on the training side, a test package,
an acceptance harness and this report.

```text
stratego/model/                        (new package, 2,187 lines)
├── __init__.py               118   public surface and the no-privileged-import statement
├── contract.py               430   versions, shapes, validated outputs, value semantics
├── tokenization.py           160   the single [B,127,10,10] -> [B,100,127] relayout
├── integration_model.py      350   integration_model_v1, the small fixture network
├── checkpoint.py             444   format, compatibility validation, loading
├── losses.py                 240   placeholder policy / value / masked-belief losses
└── policy_adapter.py         445   the reusable policy_interface_v1 neural policy

stratego/training/
└── belief_targets.py         138   privileged dense belief targets (training only)

tests/model/                           (new package, 192 tests)
├── conftest.py               107   session model, checkpoint, policies, stub policy
├── test_contract.py          196   boundary shapes, dtypes, batch consistency, value semantics
├── test_tokenization.py      124   row-major ordering pinned with position-coded tensors
├── test_policy_mapping.py    226   the exhaustive 10,000-action audit and adapter selection
├── test_legality.py          352   masking, ties, precision, non-finite, the engine guard
├── test_checkpoint.py        309   round-trip identity and every incompatibility
├── test_hidden_information.py 259  permutation trials and the object-graph audit
├── test_value_belief.py      358   value ordering/perspective, belief mask semantics
├── test_autograd.py          209   one backward pass, per-head attribution
├── test_device_batch_equivalence.py 254  CPU/MPS, float16, batch 8/64/256
└── test_evaluation_integration.py   182  the Phase 4 harness end to end

scripts/run_phase5.py        1,579   the acceptance harness that writes every artifact
checkpoints/integration_model_v1.pt   the real checkpoint the gauntlet used (untracked)
```

The tree follows the instruction's suggested layout with **one deliberate
deviation**, described in section 4.4.

### The permanent flow, as built

```text
PolicyInput (observation + legal action product, nothing privileged)
    -> observation_batch_from_numpy   [1, 127, 10, 10] float32, copied not aliased
    -> observation_to_tokens          [1, 100, 127]
    -> IntegrationModel.forward       policy [1,10000] / value [1,3] / belief [1,100,12]
    -> validate_legality              engine mask cross-checked against the legal list
    -> greedy_action | categorical_action
    -> PolicyResult
    -> match_runner._decide + validate_policy_result + apply_action  (engine is authoritative)
```

### `integration_model_v1`

| Property | Value |
|---|---|
| Width | 64 |
| Encoder blocks | 2 (pre-norm self-attention + pre-norm feed-forward, no dropout anywhere) |
| Attention heads | 4 |
| Feed-forward width | 256 |
| Parameters | **128,143** |
| Policy head | source query · destination key over 100×100, flattened row-major |
| Value head | mean-pooled tokens → 3 logits (WIN, DRAW, LOSS) |
| Belief head | per-square linear → 12 piece-type logits, sharing the encoder |
| Initialisation | deterministic, explicit CPU `torch.Generator`, seed 20250501 |
| Trained | **no** |

The blocks are written out rather than assembled from `nn.TransformerEncoder`
for two evidence-related reasons: there is no dropout anywhere, so there is no
train/eval divergence to reason about when a test claims two runs are
bit-identical; and Phase 5 forbids adopting the Phase 3 benchmark probe as the
model design, so this is an independent module rather than a re-export.

---

## 4. Contract definitions and deliberate non-decisions

### 4.1 What Phase 5 froze

| Constant | Value |
|---|---|
| `MODEL_CONTRACT_VERSION` | `model_contract_v1` |
| `MODEL_ARCHITECTURE_ID` | `integration_model_v1` |
| `ACTION_ENCODING_VERSION` | `source_destination_10000_v1` |
| `POLICY_ACTION_FRAME` | `absolute_engine_squares` |
| `TOKEN_SQUARE_FRAME` | `perspective_normalized_squares` |
| `CHECKPOINT_FORMAT_VERSION` | `1` |
| `VALUE_CLASS_ORDER` | `("WIN", "DRAW", "LOSS")` |
| `BELIEF_IGNORE_INDEX` | `-100` |
| `BELIEF_TARGET_VERSION` | `dense_belief_target_v1` |

### 4.2 The input/output frame asymmetry — stated, not resolved

The observation is **perspective-normalized**: blue's board is rotated 180
degrees (`06_observation_v2_127ch.md` section 3, implemented at
`stratego/engine/observation.py:121`). The action space is in **absolute engine
squares** (`stratego/engine/actions.py`). Phase 5 takes the literal reading of
the instruction: policy logit `a` *is* `decode_action(a)` in absolute squares,
and no remapping table exists anywhere between the model and `apply_action`, so
the engine's legality mask applies to the logits with zero transformation.

The consequence, stated plainly: a single network playing blue reads a
normalized board and must emit absolute-frame actions, so it has to learn the
180-degree flip rather than being handed it. For an untrained fixture this costs
nothing, and for Phase 5's purpose — proving the pipe connects — the literal
reading is the safer one, because every alternative introduces a transformation
between the model and the engine's legality product, which is exactly where a
silent mismatch would hide.

**This is flagged for Phase 6, not settled by Phase 5.** The engine already
ships `action_to_perspective` / `action_from_perspective`
(`stratego/engine/actions.py:45`), documented as existing "for the benefit of a
future single network that plays both colours". If Phase 6 adopts the normalized
frame, that is a new `MODEL_CONTRACT_VERSION` and a new `POLICY_ACTION_FRAME`
value; the checkpoint validator already refuses to load weights across a frame
change, so the switch cannot happen silently.

`tests/model/test_policy_mapping.py::test_a_perspective_normalized_index_is_not_what_the_adapter_uses`
pins the current choice.

### 4.3 Belief loss mask semantics

`mask[square]` is true **exactly** on squares holding a live opponent piece whose
type the acting player cannot legally know — a one-to-one correspondence with
the frozen `stratego.engine.observation.belief_target`. Excluded: own pieces,
legally revealed opponent pieces, empty squares, the two lakes, and captured
pieces. Excluded squares carry `BELIEF_IGNORE_INDEX`, the value
`cross_entropy(ignore_index=...)` skips, so a caller that forgets the mask still
cannot train on them.

Targets are indexed by **normalized** square, matching token order, because the
belief head is per token and token `i` is normalized square `i`. Indexing them
absolutely would have mis-aligned every blue-to-move target by a 180-degree
rotation — an error that would have trained silently and shown up only as a
belief head that never learns.

### 4.4 The deliberate deviation from the suggested file tree

The instruction's tree puts everything under `stratego/model/`. The privileged
dense-target builder is instead at `stratego/training/belief_targets.py`.

Reason: gate 13 asks for an object-graph and interface audit finding no
privileged products. Keeping the only module that reads `true_type` outside the
model package turns that gate from a promise into a structural assertion —
`tests/model/test_value_belief.py::test_the_model_package_never_imports_the_privileged_target_builder`
walks every module in `stratego.model` and asserts none imports `GameState`,
`PieceRecord` or `belief_target`. The model package genuinely cannot reach a
hidden type, and the test would fail the moment someone made it possible.

### 4.5 Deliberate non-decisions

Phase 5 explicitly does **not** decide:

- **the Phase 6 architecture.** `integration_model_v1` is the instruction's
  default shape, untuned. Every module, the checkpoint metadata and
  `architecture_summary()` carry the sentence "not the final/production/Ataraxos
  model";
- **loss weighting.** `DEFAULT_VALUE_WEIGHT` and `DEFAULT_BELIEF_WEIGHT` are both
  1.0 as placeholders for a single connectivity backward pass;
- **playing strength.** The fixture is untrained and its results are reported
  without being used as a gate;
- **whether belief supervision should later include revealed pieces**, or whether
  the value head should later carry more than three classes.

### 4.6 What the adapter refuses to do

Every one of these raises rather than degrading to a legal move: an empty
legality product; a dense mask that disagrees with the legal-action list; a
malformed or wrongly shaped mask; a non-finite logit on a legal action
(including a float16 overflow); an all-`-inf` row; a wrongly shaped or
non-floating policy row; a distribution with no usable probability mass. Phase 5
forbids substituting a legal action after a policy or model failure, and a
substituted move is precisely the failure that leaves no trace in any result
table.

---

## 5. Tests added and all commands run

### 5.1 Commands

```bash
python -m pytest -q                    # baseline, before any edit: 1963 passed, 2 skipped, 64.01s
python -m pytest -q                    # after implementation: 2155 passed, 2 skipped, 104.10s
python scripts/run_phase5.py --quick   # smoke run of the harness
python scripts/run_phase5.py           # full acceptance run, 736.39s, status PASS
```

### 5.2 New tests, per module

| Module | Tests | Result | What it proves |
|---|---:|---|---|
| `test_contract.py` | 22 | pass | canonical `[B,127,10,10]` boundary, dtype family, batch consistency, WIN/DRAW/LOSS semantics |
| `test_tokenization.py` | 11 | pass | row-major ordering pinned with position-coded tensors, exact inverse, read-only input copied |
| `test_policy_mapping.py` | 10 | pass | all 10,000 identifiers round-trip; crafted maxima select the same engine action |
| `test_legality.py` | 36 | pass | masking, ties, extreme/float16/non-finite logits, malformed masks, the sampler regression, the engine guard |
| `test_checkpoint.py` | 45 | pass | bit-identical reload, every missing/wrong/corrupt case |
| `test_hidden_information.py` | 9 | pass | permutation trials plus the object-graph and interface audit |
| `test_value_belief.py` | 22 | pass | value ordering and acting-player perspective, belief mask semantics, label/input separation |
| `test_autograd.py` | 8 | pass | one backward pass, per-head gradient attribution |
| `test_device_batch_equivalence.py` | 17 | pass | CPU/MPS float32 and float16, batch 8/64/256 |
| `test_evaluation_integration.py` | 12 | pass | the Phase 4 harness end to end, both modes |
| **Total** | **192** | **all pass** | |

---

## 6. Exhaustive action and legality results

### 6.1 The 10,000-action audit

| Check | Count | Mismatches |
|---|---:|---:|
| `decode -> (source, destination) -> encode` over the whole space | 10,000 | **0** |
| `action_id == 100 * source + destination` and the accessor helpers | 10,000 | **0** |
| Distinct `(source, destination)` pairs (bijection onto 100×100) | 10,000 | **0** |
| Tokenization position checks (2 batches × 100 squares) | 200 | **0** |
| Tokenization inverse exact | — | exact |

### 6.2 Adapter selection

A crafted unique maximum was placed on **every legal action** of a 12-position
corpus and the adapter was required to select that engine action: **264
selections, 0 mismatches.** Move families covered: attack, quiet, single step,
long Scout slide, lateral, vertical. The same claim is re-checked through the
full `decide()` path — requirement plumbing, legality cross-check, diagnostics —
and through `apply_action`, confirming the piece named by the identifier's
source square actually moves to (or dies attacking) its destination square.

### 6.3 Legality and numerical edge cases

All required cases are covered in `tests/model/test_legality.py` (36 tests):

| Case | Behaviour |
|---|---|
| Highest raw logit is illegal | ignored, best legal action chosen |
| All largest raw logits are illegal | ignored, best legal action chosen |
| Exactly one legal action | that action, even at logit −1e30 |
| Tied legal maxima | lowest action identifier, independent of input order |
| Extreme finite logits (±`float32.max`) | usable, sampling normalises correctly |
| float16 logits | widened to float32 for comparison, selection correct |
| float16 overflow on a legal action | **raises** — not silently widened |
| `NaN` / `+inf` / `−inf` on a legal action | **raises** |
| `NaN` / `+inf` / `−inf` on an illegal action | ignored, never read |
| All-`−inf` row | **raises** |
| Empty legal set, empty mask | **raises** |
| Malformed mask (wrong length, wrong rank, values ∉ {0,1}) | **raises** |
| Mask disagrees with the legal-action list | **raises** |
| Duplicate / out-of-range action identifiers | **raises** |

**The Phase 3 Gumbel regression is permanent.** Phase 3 lost a run to a
Gumbel-max sampler whose uniform draw could be exactly zero → `+inf` noise →
`NaN` after adding the `−inf` illegal fill → `argmax` ranks `NaN` first → an
illegal action was chosen. The Phase 5 sampler is deliberately **not**
Gumbel-max: one `rng.random()` walked along an explicit float64 cumulative sum,
indexing into the legal list, so an extreme draw cannot name an illegal action
at all. Tested at draws 0.0, 1e-300, 0.5, `1 − 2⁻⁵³` and 0.9999999999999999,
plus a case where all but one probability underflows to exactly zero. The Phase
3 `_gumbel_noise` guard itself is also re-asserted so the original fix cannot
regress underneath us.

### 6.4 The engine's illegal-action guard

An illegal action applied directly to the engine raises `IllegalActionError`
and leaves the state **completely inert** — verified by comparing a full
snapshot including history before and after: no piece moved, no ply counted, no
event emitted. A forged `PolicyResult` naming an illegal action is caught by
`validate_policy_result`, and the match runner classifies it as
`illegal_action` rather than swallowing it.

---

## 7. Hidden-information audit

**10,000 valid paired trials. 0 mismatches. 0 positive-control failures.**

| Field | Value |
|---|---|
| Trials | 10,000 (target 10,000) |
| Source positions | 500 seeded random games |
| Plies sampled | 15, 30, 55, 85, 125, 180, 240 |
| Trials per ply | 1,440 / 1,440 / 1,440 / 1,420 / 1,420 / 1,420 / 1,420 |
| Hidden pieces permuted per trial | min 17, mean 32.0, max 40 |
| Skipped: invalid | 0 |
| Skipped: unchanged permutation | 0 |
| Skipped: fewer than two hidden pieces | 0 |
| Permutation seed | 90210 |
| Inference | deterministic CPU, fixed weights |
| Runtime | 28.2 s |

Every trial takes a real position, permutes the true types of the opponent
pieces the acting player cannot legally know, and requires that everything the
model sees and everything it says is identical:

| Requirement | Mismatches |
|---|---:|
| Observations identical | **0** |
| Legal actions identical | **0** |
| Policy logits identical | **0** |
| Value logits identical | **0** |
| Belief logits identical | **0** |
| Adapter greedy action identical | **0** |
| Adapter diagnostics identical | **0** |

Positive controls, which must fire on **every** accepted trial:

| Control | Failures |
|---|---:|
| Privileged belief target differs after the permutation | **0** |
| Underlying permuted hidden types actually differ | **0** |

### 7.1 Structural audit — nothing privileged is reachable

Beyond the trials, three structural checks:

1. **Requirements.** Both policy modes declare exactly `observation=True` and
   `legal_action_mask=True`; `public_view`, `public_events` and `public_setup`
   are all false, so those products are never even built.
2. **Object graph.** A traversal of the live adapter — attributes, containers,
   dicts, tensors — finds **no** `GameState` and **no** `PieceRecord`.
3. **Import graph.** No module in `stratego.model` imports `GameState`,
   `PieceRecord`, `belief_target`, or `stratego.training.belief_targets`.

The observation and mask handed to the model are read-only NumPy arrays, and
`observation_batch_from_numpy` copies rather than aliasing, so the model cannot
write back into engine-owned memory. A request built without the observation
raises `PolicyContractError` instead of silently proceeding.

---

## 8. Checkpoint compatibility and save/load results

### 8.1 Round-trip identity on CPU

| Check | Result |
|---|---|
| Policy logits bit-identical after save → destroy → reload | **yes** |
| Value logits bit-identical | **yes** |
| Belief logits bit-identical | **yes** |
| Greedy action identical | **yes** |
| `state_dict` digest stable across the round trip | **yes** |

```text
checkpoint    checkpoints/integration_model_v1.pt   (526,910 bytes)
file digest   268c0a27b20218be0cd6c10539b561916e238733954a953b8eea6812a1eb35de
weights       5b3a8dda202a7041d7241db9bae965ffce53b2ae30ba281cc2351ad3c8825768
```

### 8.2 Stored fields

`checkpoint_format_version`, `model_architecture_id`, `model_contract_version`,
`rules_version`, `observation_version`, `action_encoding_version`,
`policy_action_frame`, `model_configuration`, `state_dict`,
`training_iteration`, `training_step`, `creation_timestamp`, optional
`optimizer_state`, optional `ema_state`, optional `training_metrics`, and a
`provenance` block (torch version, initialisation seed, parameter count, weights
digest, and the fixture warning).

Weights are always stored as CPU float32, so a checkpoint written from a Metal
or float16 run reloads identically anywhere; precision is a run-time choice.
Files are written to a temporary sibling and renamed, so an interrupted save
cannot leave a half-written file. Loading uses `weights_only=True`, so a
tampered file cannot execute code.

### 8.3 Every incompatibility fails loudly — 24 / 24 rejected

| Category | Cases | Rejected |
|---|---:|---:|
| Missing required metadata (`rules_version`, `state_dict`, `model_configuration`, `creation_timestamp`) | 4 | 4 |
| Unknown / newer format version, zero version | 2 | 2 |
| Wrong rules, superseded observation, wrong action encoding, wrong model contract, wrong architecture, wrong policy frame | 6 | 6 |
| Incompatible or unknown configuration field | 2 | 2 |
| Unknown top-level field | 1 | 1 |
| Missing / unexpected / wrongly shaped weight | 3 | 3 |
| Negative training counter | 1 | 1 |
| Truncated file | 1 | 1 |
| Random-bytes file | 1 | 1 |
| Empty file | 1 | 1 |
| Bare `state_dict` file (`torch.save(model.state_dict(), path)`) | 1 | 1 |
| Missing file | 1 | 1 |
| **Total** | **24** | **24** |

Semantics are checked **before** shapes, so a checkpoint saved under
`observation_v2_127ch` is reported as an observation-version mismatch rather than
as a confusing weight error — and it is never loaded "because the tensors fit".
There is no override switch.

---

## 9. Value, belief and autograd results

### 9.1 Value

| Check | Result |
|---|---|
| Class order | `WIN`, `DRAW`, `LOSS` |
| Probabilities sum to one | yes |
| `E[v] = P_W − P_L` | matches to 1e-7 |
| Win-heavy controlled row | `E[v] = +0.997515` |
| Loss-heavy controlled row | `E[v] = −0.997515` |
| Certain-draw controlled row | `E[v] = 0.000000` |
| Acting-player perspective, both colours | 2 mirrored pairs, **0 mismatches** |

The perspective check uses a game and its colour-swapped, 180-degree-rotated
twin. Red-to-move in one and blue-to-move in the other produce byte-identical
observations and therefore identical value logits — which is only correct if the
head means "the acting player's chances" rather than "red's chances". A scalar
value head is rejected at the boundary.

### 9.2 Belief

| Check | Result |
|---|---|
| Logits shape | `[B, 100, 12]`, finite |
| Supervised squares per position (8-position corpus) | min 24, mean 32.4, max 40 |
| Excluded squares per position | mean 67.6 |
| Own pieces supervised | **0 violations** |
| Empty squares supervised | **0 violations** |
| Lakes supervised | **0 violations** |
| Legally revealed opponent pieces supervised | **0 violations** |
| Label / mask disagreements | **0** |
| Disagreements with the frozen sparse `belief_target` | **0** |

The masked loss is proven to ignore unsupervised squares in two independent
ways: a backward pass leaves gradients at exactly zero on every unsupervised
square, and changing an unsupervised square's logits by 1000 does not change the
loss at all. Normalisation is per supervised square. A mask that disagrees with
the labels raises. Labels are proven separate from inputs: permuting hidden
identities changes the target and leaves the observation byte-identical.

### 9.3 Autograd connectivity — one controlled backward pass

`L = L_policy + 1.0·L_value + 1.0·L_belief`, one backward pass, **zero optimizer
steps**.

| Component | Value |
|---|---|
| Policy loss | 2.6477 |
| Value loss | 1.1226 |
| Belief loss | 2.8662 |
| **Total** | **6.6365** |
| All components finite | yes |
| Parameters with no gradient | **none** |
| Parameters with a non-finite gradient | **none** |

| Group | Tensors | Max gradient norm | All non-zero |
|---|---:|---:|---|
| Shared encoder | 29 | 6.4391 | yes |
| Policy head | 4 | 0.5594 | yes |
| Value head | 4 | 0.4980 | yes |
| Belief head | 2 | 2.6639 | yes |

Attribution is unambiguous because each head is also driven by its **own**
backward pass in isolation: each one moves its own parameters and the shared
encoder, and leaves the other two heads at exactly zero.

The masked policy loss uses a large finite penalty rather than `-inf`. Inference
masks with true `-inf`, which is exact and harmless because nothing
differentiates it; in a loss, `-inf` makes `log_softmax` produce `NaN` gradients
as soon as a row is dominated by masked entries. The substitution is numerically
negligible (`exp(-1e9)` underflows to exactly zero in float32) and keeps every
gradient finite — verified on a row with exactly one legal action, where no
gradient escapes onto an illegal index.

---

## 10. CPU / MPS, precision, batch and performance results

### 10.1 Predeclared tolerances

| Comparison | atol | rtol |
|---|---|---|
| float32 across devices | 1e-4 | 1e-4 |
| float16 | 5e-2 | 5e-2 |

These are the instruction's starting policy, used **unchanged**. They are
declared in both `tests/model/test_device_batch_equivalence.py` and
`scripts/run_phase5.py`.

### 10.2 Measured cross-device error, per head

| Configuration | Head | Max absolute error | Max relative error |
|---|---|---:|---:|
| MPS float32 | policy logits | 7.15e-07 | 3.06e-03 |
| MPS float32 | value probabilities | 5.96e-08 | 2.47e-07 |
| MPS float32 | belief logits | 8.34e-07 | 8.00e-04 |
| MPS float16 | policy logits | 1.60e-03 | 1.54e+01 |
| MPS float16 | value probabilities | 8.72e-05 | 2.94e-04 |
| MPS float16 | belief logits | 1.75e-03 | 1.26e+01 |

**Read the relative-error column carefully.** It is large for the float16 logit
heads because relative error is meaningless where a logit is near zero — a
1.6e-03 absolute difference on a logit of 1e-04 is a relative error of 15. The
comparison passes because `allclose` is `atol + rtol·|b|` and `atol` dominates in
exactly that regime. Absolute error is the honest number here, and it is 1.75e-03
at worst in float16 and 8.34e-07 in float32. Value **probabilities**, which are
bounded and not near zero, agree to 2.94e-04 relative even in float16.

Both configurations produced **finite outputs everywhere**.

### 10.3 Greedy agreement

| Configuration | Crafted-margin agreement | Natural-corpus agreement |
|---|---|---|
| MPS float32 | **48 / 48 exact** | 8 / 8 |
| MPS float16 | **48 / 48 exact** | 8 / 8 |

Exact agreement is *asserted* only on crafted-margin examples, where one legal
action leads by 100 logits — far more than any tolerance. Natural-corpus
agreement is measured and reported separately rather than asserted exactly,
because an untrained network produces near-ties that a kernel difference can
legitimately reorder; hiding that behind a wide tolerance would be the dishonest
version. On this corpus it happened to be perfect at both precisions.

Legal-action sets are identical across devices by construction: legality is an
engine product and no device participates in producing it.

### 10.4 Batch equivalence

Same position alone versus embedded in a larger batch, on CPU:

| Batch | Policy logits | Value logits | Belief logits | Selected action |
|---|---:|---:|---:|---|
| 8 | 0.0 | 1.19e-07 | 0.0 | identical |
| 64 | 0.0 | 1.19e-07 | 0.0 | identical |
| 256 | 0.0 | 1.19e-07 | 0.0 | identical |

Policy and belief are **bit-exact**; the value head differs by one float32 ulp
because mean-pooling reduces over a differently shaped tensor. All within
tolerance. The same comparison on MPS float32 also passes at every batch size.

### 10.5 Minimal performance baseline

MPS, after 5 warmup iterations, `torch.mps.synchronize()` between timings.
Tokenization and masking are **outside** the timed region for the MPS rows and
**inside** it for the CPU decision row, which is stated per row in the artifact.

| Device | Precision | Batch | Median latency | Positions/s | OOM |
|---|---|---:|---:|---:|---|
| MPS | float32 | 1 | 1.012 ms | 988 | no |
| MPS | float32 | 64 | 3.220 ms | 19,877 | no |
| MPS | float32 | 256 | 10.195 ms | 25,110 | no |
| MPS | float32 | 1,024 | 38.341 ms | 26,708 | no |
| MPS | float16 | 1 | 1.347 ms | 742 | no |
| MPS | float16 | 64 | 2.603 ms | 24,589 | no |
| MPS | float16 | 256 | 9.109 ms | 28,105 | no |
| MPS | float16 | 1,024 | 34.857 ms | 29,377 | no |
| CPU | float32 | 1 | 0.675 ms | 1,482 | no |

Batch 1,024 did **not** run out of memory at either precision. At batch 1 the
CPU is faster than Metal, as expected — a 128k-parameter forward pass does not
amortise the dispatch overhead. Throughput saturates around 25–29k positions/s
by batch 256.

**These numbers are evidence, not a threshold, and nothing was tuned against
them.** Section 5.7 reserves the architecture sweep for Phase 6; they are also
measured on a 128k-parameter fixture and say essentially nothing about what a
real model will cost.

---

## 11. Phase 4 gauntlet and reproduction results

A real `integration_model_v1` checkpoint was saved, reloaded through the
reusable adapter, and run against all four accepted core baselines in both
decision modes: **64 paired units per matchup**, `color_swap_same_board`, the
accepted setup bank, MatchSpec, seeds and runner unchanged.

```text
1,024 matches   603,874 plies   601.1 s   worker_count = 1
```

### 11.1 Integrity — the part that is a gate

| Requirement | greedy | seeded categorical |
|---|---|---|
| Illegal actions | **0** | **0** |
| Policy failures | **0** | **0** |
| Colour swap correct (both colours, same board, per unit) | **yes** (256 units) | **yes** (256 units) |
| Full rerun identical (`compare_results`) | **yes**, digest `203d1608ded44b0b` | **yes**, digest `9a8351da7f75a3cf` |
| Engine replay of stored action histories | 32 sampled, **0 problems** | 32 sampled, **0 problems** |
| Reproduction from a stored row alone, without the bank | 32 sampled, **0 differences** | 32 sampled, **0 differences** |
| Checkpoint / model / version identity in results | yes | yes |

The two modes produce different result digests, confirming the stochastic gate
is not accidentally re-testing the greedy path.

### 11.2 Playing strength — reported, **not** a gate

The fixture is untrained. These numbers describe random weights.

**Greedy (`integration_model_v1_greedy@0.1.0`)**

| Opponent | W | D | L | EWR | Mean plies | Flag capture | Move-limit draw |
|---|---:|---:|---:|---:|---:|---:|---:|
| `random_legal@1.0.0` | 8 | 108 | 12 | 0.484 | 1,337 | 9.4% | 84.4% |
| `basic_heuristic@1.0.0` | 6 | 16 | 106 | 0.109 | 371 | 80.5% | 12.5% |
| `tactical_rule_based@1.0.0` | 0 | 32 | 96 | 0.125 | 373 | 71.9% | 25.0% |
| `strategic_rule_based@1.1.0` | 1 | 30 | 97 | 0.125 | 401 | 75.0% | 23.4% |

**Seeded categorical (`integration_model_v1_sampled@0.1.0`)**

| Opponent | W | D | L | EWR | Mean plies | Flag capture | Move-limit draw |
|---|---:|---:|---:|---:|---:|---:|---:|
| `random_legal@1.0.0` | 33 | 42 | 53 | 0.422 | 1,312 | 15.6% | 32.0% |
| `basic_heuristic@1.0.0` | 14 | 3 | 111 | 0.121 | 334 | 73.4% | 2.3% |
| `tactical_rule_based@1.0.0` | 4 | 0 | 124 | 0.031 | 296 | 90.6% | 0.0% |
| `strategic_rule_based@1.1.0` | 2 | 0 | 126 | 0.016 | 294 | 97.7% | 0.0% |

Each row is 128 games (64 paired units). Read plainly: the untrained fixture is
roughly even with `random_legal` — mostly by grinding to the battleless move
limit — and loses heavily to all three rule-based baselines, mostly by having
its flag captured. That is what a random policy should do. **No conclusion about
the architecture may be drawn from these numbers**, and none is drawn here.

One observation worth carrying to Phase 6: the greedy mode draws far more than
the sampled mode (108 vs 42 draws against `random_legal`) because a deterministic
argmax over fixed random weights repeats moves until the battleless limit fires.
It is a property of untrained determinism, not of the adapter.

### 11.3 Why the gauntlet runs serially

`run_schedule(worker_count > 1)` rebuilds policies from the Phase 4 catalogue
inside each worker process, and the neural policy is deliberately **not** in
that catalogue, so a passed-in torch model cannot cross the process boundary.
Registering the fixture in the catalogue would have changed what every Phase 4
audit that enumerates "all policies" means, so the gauntlet runs at
`worker_count=1` instead. At 128k parameters this costs 601 s for 1,024 matches
and is not a problem; at Phase 6 scale it will be, which is noted in the handoff.

---

## 12. Completion-gate table with evidence locations

| # | Gate | Result | Evidence |
|---:|---|---|---|
| 1 | `frozen_contracts_verified_unchanged` | **true** | `agent_01_phase5_acceptance.json` → `frozen_contracts`; report §2.3 |
| 2 | `preexisting_suite_green` | **true** | `agent_01_phase5_acceptance.json` → `preexisting_suite`; report §2.1 |
| 3 | `full_suite_green_after_changes` | **true** | `test_suite` → 2,155 passed, 0 failed |
| 4 | `input_shape_and_dtype_validated` | **true** | `tests/model/test_contract.py` (22 pass) |
| 5 | `tokenization_exact_row_major` | **true** | `agent_01_action_mapping.json` → 0 mismatches; `test_tokenization.py` (11 pass) |
| 6 | `policy_output_contract_validated` | **true** | `agent_01_action_mapping.json` → `bijection_ok` |
| 7 | `value_output_contract_validated` | **true** | `agent_01_value_belief_autograd.json` → `value`; §9.1 |
| 8 | `belief_output_and_mask_validated` | **true** | same file → `belief.exclusion_violations` all zero; §9.2 |
| 9 | `all_10000_actions_round_trip` | **true** | `agent_01_action_mapping.json` → 10,000 checked, 0 mismatches |
| 10 | `policy_index_matches_engine_action` | **true** | same file → 264 crafted selections, 0 mismatches |
| 11 | `legality_edge_cases_pass` | **true** | `tests/model/test_legality.py` (36 pass); §6.3 |
| 12 | `engine_illegal_action_guard_preserved` | **true** | `agent_01_action_mapping.json` → guard raised **and** inert |
| 13 | `no_privileged_input_reachable` | **true** | `tests/model/test_hidden_information.py` (9 pass); §7.1 |
| 14 | `hidden_information_10000_zero_mismatch` | **true** | `agent_01_hidden_information.json` → 10,000 trials, 0 mismatches, 0 control failures |
| 15 | `checkpoint_cpu_roundtrip_identity` | **true** | `agent_01_checkpoint_compatibility.json` → `round_trip_identity` |
| 16 | `checkpoint_incompatibilities_fail_loudly` | **true** | same file → 19 negatives + 5 corrupt files, all rejected |
| 17 | `greedy_and_seeded_modes_reproducible` | **true** | `agent_01_evaluation_gauntlet.json` → both `rerun_identical` |
| 18 | `autograd_all_heads_connected_finite` | **true** | `agent_01_value_belief_autograd.json` → `autograd`; §9.3 |
| 19 | `cpu_mps_float32_equivalence_pass` | **true** | `agent_01_numerical_batch_performance.json` → `devices.mps_float32` |
| 20 | `mps_float16_finite_and_equivalent` | **true** | same file → `devices.mps_float16` |
| 21 | `batch_equivalence_pass` | **true** | same file → `batch_equivalence` |
| 22 | `phase4_gauntlet_pass` | **true** | `agent_01_evaluation_gauntlet.json` / `.csv` → `all_clean` |

**22 / 22 true.**

### Artifacts

| Path | Contents |
|---|---|
| `reports/phase_5_data/agent_01_phase5_acceptance.json` | the 22 gates, environment, headline numbers, suite results |
| `reports/phase_5_data/agent_01_action_mapping.json` | the exhaustive action audit and tokenization ordering |
| `reports/phase_5_data/agent_01_hidden_information.json` | the 10,000-trial model-level audit |
| `reports/phase_5_data/agent_01_checkpoint_compatibility.json` | round-trip identity, 24 rejection cases, digests |
| `reports/phase_5_data/agent_01_numerical_batch_performance.json` | device, precision, batch and latency measurements |
| `reports/phase_5_data/agent_01_value_belief_autograd.json` | value semantics, belief mask audit, gradient attribution |
| `reports/phase_5_data/agent_01_evaluation_gauntlet.json` | per-mode, per-opponent gauntlet summary and reproduction |
| `reports/phase_5_data/agent_01_evaluation_gauntlet.csv` | all 1,024 matches in the Phase 4 row format |
| `checkpoints/integration_model_v1.pt` | the checkpoint the gauntlet used (untracked; digest recorded) |

Every headline number in this document also exists in those files.

---

## 13. Known limitations and deviations

1. **The fixture is untrained and loses.** `integration_model_v1` has random
   weights. It loses to all three rule-based baselines by flag capture and
   grinds `random_legal` to the move limits. This is the expected behaviour of a
   random policy and is reported, not used as a gate — Phase 5's claim is that
   the pipe carries a real checkpoint correctly, not that the checkpoint plays.

2. **File-tree deviation.** The privileged dense belief-target builder lives at
   `stratego/training/belief_targets.py` rather than under `stratego/model/`.
   Rationale in §4.4; it strengthens gate 13 rather than weakening it.

3. **The gauntlet runs serially.** See §11.3. Phase 6 will want a
   checkpoint-aware worker initializer; that is a Phase 6 design choice, not a
   Phase 5 repair.

4. **Natural-corpus greedy agreement is reported, not asserted exactly.** See
   §10.3. It was 8/8 at both precisions here, but the test asserts a floor of
   0.5 rather than exactness, on purpose.

5. **The frame asymmetry is unresolved by design.** See §4.2. It is recorded in
   the contract and the checkpoint metadata, and refusing to load across a frame
   change is tested, but which frame the final model should use belongs to
   Phase 6.

6. **Adapter diagnostics are not persisted into results.** The adapter reports
   value probabilities and the expected value per decision, but `MatchResult`
   does not carry per-decision diagnostics, so those values appear in tests and
   in the hidden-information audit rather than in the gauntlet CSV.

7. **Replay and row-only reproduction were sampled, not exhaustive.** 32 of 512
   rows per mode were replayed through the engine and reproduced from the stored
   row alone (0 problems). The *full* rerun comparison, however, covers all 512
   matches per mode and is digest-identical, so every match is covered by at
   least one reproduction check.

8. **Nothing is committed.** All Phase 5 work is left in the working tree, as
   requested. The checkpoint is at `checkpoints/integration_model_v1.pt`, which
   is untracked and not listed in `.gitignore`; its SHA-256 is recorded in
   `agent_01_checkpoint_compatibility.json` so the exact weights the gauntlet
   used are identifiable.

---

## 14. Exact handoff recommendation for Phase 6

**Keep, unchanged:**

- `stratego/model/contract.py` — the four shapes, the WIN/DRAW/LOSS ordering,
  `E[v] = P_W − P_L`, and the boundary validators. This is the part of Phase 5
  intended to outlive it.
- `stratego/model/tokenization.py` — the relayout is exact and pinned.
- `stratego/model/checkpoint.py` — the format already carries every field a
  training loop needs, including optional optimizer and EMA state, and refuses
  every incompatibility.
- `stratego/model/policy_adapter.py` — the selection rules, the tie-break, the
  single-draw seed contract and the refusal behaviour.
- `stratego/training/belief_targets.py` — and its position outside the model
  package.

**Replace:**

- `stratego/model/integration_model.py`. It is a fixture. Phase 6 picks the real
  architecture and gives it a new `MODEL_ARCHITECTURE_ID`; the checkpoint
  validator will then refuse to load fixture weights into it, which is the
  intended behaviour rather than an obstacle.
- `stratego/model/losses.py`. The three losses are placeholders with weight 1.0
  and no schedule.

**Decide before any training run:**

1. **The policy action frame** (§4.2). Training under `absolute_engine_squares`
   and later switching means discarding the weights. Decide it, then bump
   `MODEL_CONTRACT_VERSION` if it changes.
2. **Whether belief supervision covers revealed opponent pieces** (§4.3).
3. **Checkpoint-aware parallel evaluation** (§11.3), which is what makes a
   full-size gauntlet affordable once the model is larger than 128k parameters.

**Do not re-derive:** the 10,000-action audit, the tokenization ordering, the
hidden-information trial construction and the checkpoint negative-case list are
all in place and cheap to re-run — `python scripts/run_phase5.py` regenerates
every artifact in this report in 736 s.

---

*Phase 5 status recommendation: **PASS**. Phase 5 is not complete merely because
the code runs; the reviewing chat must inspect the evidence above and formally
accept the phase before Phase 6 begins.*
