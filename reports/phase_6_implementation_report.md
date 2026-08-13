# Phase 6 Implementation Report

Production model architecture selection and M4 Pro benchmarking.

Frozen throughout Phase 6: rules `stratego_project_v1`, reference engine
`phase2_1_reference_1.1.0`, observation `observation_v2_1_127ch`, engine action
encoding `source_destination_10000_v1` in absolute engine squares, Phase 3
backend `KEEP_PYTHON`, and Phase 4 match/evaluation semantics.

## 1. Agent 1 — Model Contract v2 and Perspective-Normalized Actions

**Status: PASS** — 16 / 16 completion gates true.

### 1.1 Prerequisite verification

Verified from the repository rather than assumed, by reading
`reports/phase_5_data/agent_01_phase5_acceptance.json`:

| Check | Required | Found |
|---|---|---|
| Phase 5 status | `PASS` | `PASS` |
| Phase 5 hard gates | 22 / 22 | 22 / 22 |
| Phase 5 quick mode | false | false |
| `model_contract_v1` present | yes | yes, at commit `1d8e7cb` |
| `integration_model_v1` integration-only | yes | yes (`MODEL_ARCHITECTURE_ID` + fixture note) |

Pre-existing suite, measured at commit `8f4f5e3` **before any Phase 6 edit**:

```text
python -m pytest -q
2155 passed, 2 skipped, 0 failed in 102.29s
```

That is identical to the totals recorded by the accepted Phase 5 run, so the
starting tree was green and unmodified. The two skips are the pre-existing
Phase 4 capability skips (`random_legal` and `stress_chaos` expose no per-move
score vector); they are unrelated to Phase 6 and were not repaired.

### 1.2 What changed

`model_contract_v2` moves the **model-facing** action space into the acting
player's perspective-normalized squares. The engine is untouched.

```text
TOKEN_SQUARE_FRAME     perspective_normalized_squares
POLICY_ACTION_FRAME    perspective_normalized_squares   (was absolute_engine_squares)
ENGINE_ACTION_FRAME    absolute_engine_squares          (new, explicit)
ACTION_ENCODING        source_destination_10000_v1      (unchanged)
```

Under v1 the network read a board that had already been rotated for blue and
then had to emit an action in unrotated squares, so every strategic concept had
to be learned once per colour. Under v2 the tokens and the policy logits share
one frame and one weight means one strategic move regardless of colour.

**One conversion, one place.** `stratego/model/action_frame.py` is the only
module that converts between frames, and it does not re-derive the geometry: its
tables are built by calling the frozen engine helpers
`stratego.engine.actions.action_to_perspective` / `action_from_perspective` for
all 10,000 identifiers. Import-time checks prove each table is a permutation of
`0..9999` and that the two directions compose to the identity both ways, so a
non-bijective build cannot start. No second coordinate convention was created.

**The decision path.**

```text
normalized observation
  -> model logits in the normalized action frame
  -> engine legality (list + dense mask), converted into that frame
  -> greedy or seeded categorical selection, entirely in the model frame
  -> model_action_to_absolute(...)
  -> PolicyResult carrying the absolute engine action
  -> Phase 4 validation -> engine apply_action
```

Preserved exactly: deterministic greedy tie-break, one random stream per
decision drawn from once, seeded categorical reproducibility, non-finite
rejection, malformed/empty legality rejection, no action substitution after a
failure, and independent engine validation of the result.

One deliberate semantic consequence: the greedy tie-break ("lowest identifier
among the maximal logits") is now resolved on the **normalized** identifier. For
blue that selects a different move than v1 would have. This is the intended
behaviour — a tie-break that depended on colour would reintroduce exactly the
asymmetry v2 removes — and it is why the policy identity moved (§1.5).

### 1.3 Files

Created:

| File | Purpose |
|---|---|
| `stratego/model/action_frame.py` | the single authoritative frame conversion |
| `tests/model/test_action_frame.py` | exhaustive bijection, pinned geometry, legality equivalence |
| `tests/model/test_symmetry.py` | colour-symmetry regression and its negative control |
| `scripts/run_phase6_agent01.py` | this agent's acceptance harness |
| `checkpoints/integration_model_v2.pt` | v2 fixture checkpoint |
| `reports/phase_6_data/agent_01_model_contract_v2.json` | machine-readable results |
| `reports/phase_6_implementation_report.md` | this report |

Modified:

| File | Change |
|---|---|
| `stratego/model/contract.py` | `model_contract_v2`, new `ENGINE_ACTION_FRAME`, frozen `LEGACY_CONTRACT_V1` record |
| `stratego/model/checkpoint.py` | both frames required and checked first, v1 recognition, `accepted_under_contract_v1` |
| `stratego/model/policy_adapter.py` | normalized decision path, policy ids and version bumped |
| `stratego/model/__init__.py` | export the new surface |
| `tests/model/conftest.py` | `repository_root` fixture, stub policy id |
| `tests/model/test_contract.py` | pin v2 frames and the version move |
| `tests/model/test_checkpoint.py` | v2 metadata, retargeted negatives, new v1/v2 boundary section |
| `tests/model/test_evaluation_integration.py` | v2 policy identity |
| `scripts/run_phase5.py` | v2 checkpoint path, retargeted negatives, `--data-directory` |

`stratego/engine/` was **not** modified. No Phase 4 identity, statistic, setup
bank, pairing rule or result schema was changed.

### 1.4 Measured results

Command: `python scripts/run_phase6_agent01.py` — 292.28 s total, status `PASS`,
16 / 16 gates, on macOS 26.5.2 / arm64, Python 3.13.2, torch 2.13.0
(MPS built and available; this agent's work is CPU-only and requires no MPS
measurement).

**Action frame — exhaustive, not sampled**

| Measurement | Result |
|---|---|
| absolute → model → absolute cases | 20,000 (10,000 actions × 2 players) |
| absolute round-trip mismatches | **0** |
| model → absolute → model cases | 20,000 |
| reverse round-trip mismatches | **0** |
| collisions | **0** |
| full bijection over `0..9999` | true, both players |
| action encoding preserved (`100·s + d`) | true |
| red transform is the identity | true |
| blue identifiers moved | 10,000 of 10,000 |
| pinned geometry cases | 11, **0** mismatches |
| duration | 0.031 s |

Blue has no fixed points at all: `99 - s == s` has no integer solution on a
100-square board, so the transform moves every one of the 10,000 identifiers.
An adapter that silently forgot to convert for blue has nowhere to hide.

The 11 pinned cases are written out by hand and cover the first and last
squares, the first and last rows and columns, lateral and vertical single steps,
full-row and full-column Scout runs, a long Scout run past a lake column, and
corner-to-corner. Example: `d4->d7` for blue reads as `g7->g4`.

**Legal-action and dense-mask equivalence over real positions**

| Measurement | Result |
|---|---|
| positions tested | 16 (real non-terminal engine positions) |
| both colours acting | true |
| legal-action comparisons | 356 |
| transformed list ≠ nonzero transformed-mask indices | **0** |
| normalized set → absolute ≠ engine set | **0** |
| dense mask round-trip differences | **0** |
| mask dtype changes | 0 |
| legal actions per position | min 6, mean 22.25, max 37 (observed, not assumed) |
| duration | 0.006 s |

No maximum legal-action count is hard-coded anywhere; the range above is
reported, not enforced.

**Symmetry regression**

| Measurement | Result |
|---|---|
| mirrored position pairs compared | 200 |
| normalized observation mismatches | **0** |
| normalized legal-mask mismatches | **0** |
| normalized legal-set mismatches | **0** |
| normalized chosen-action mismatches | **0** |
| absolute actions not mirror images | **0** |
| v1 absolute-frame control disagreements | **189 of 200** |
| terminal pairs skipped (seed advanced instead) | 11 |
| duration | 0.668 s |

The last row is the point of the exercise: under the retired v1 rule the same
network, on the same two equivalent positions, chose non-corresponding moves in
189 of 200 cases. Under v2 it chose the same normalized move in 200 of 200.

**Checkpoint compatibility**

| Measurement | Result |
|---|---|
| compatibility cases | 11 |
| cases behaving as expected | 11 |
| rejection failures | **0** |
| real `integration_model_v1.pt` rejected | true |
| v1 file unmodified by the refused load | true (digest `268c0a27…` before and after) |
| v2 payload refused under the frozen v1 rule | true |
| v1 rule still accepts a v1 payload | true (so the check is not vacuous) |
| duration | 0.017 s |

Cases: the shipped v1 file, a synthetic v1 payload, wrong `policy_action_frame`
(two values), wrong `engine_action_frame`, wrong `model_contract_version` (v1
and v3), each frame field missing, and the two positive cases (current v2
payload and current v2 file).

**Hidden-information audit under v2**

| Measurement | Result |
|---|---|
| valid permutation trials | 10,000 |
| source positions | 500, plies 15–240 |
| hidden pieces permuted per trial | min 17, mean 32.0, max 40 |
| trials skipped (invalid / unchanged / too few hidden) | 0 / 0 / 0 |
| normalized observation mismatches | **0** |
| normalized legal-set mismatches | **0** |
| normalized legal-mask mismatches | **0** |
| policy / value / belief logit mismatches | **0** / **0** / **0** |
| model-frame chosen action mismatches | **0** |
| absolute engine chosen action mismatches | **0** |
| public diagnostics mismatches | **0** |
| **total mismatches** | **0** |
| positive control — belief targets changed | 0 failures |
| positive control — hidden true types changed | 0 failures |
| duration | 29.9 s |

The two products v2 adds — the normalized legality set/mask and the model-frame
action recorded in diagnostics — are audited alongside the Phase 5 products, so
the migration widened this audit rather than inheriting it.

**Phase 4 integration regression**

| Measurement | Greedy | Seeded categorical |
|---|---|---|
| policy | `integration_model_v2_greedy@0.2.0` | `integration_model_v2_sampled@0.2.0` |
| matches | 128 | 128 |
| paired units | 64 | 64 |
| colours played | red and blue | red and blue |
| colour-swap pairing correct | true | true |
| illegal actions | **0** | **0** |
| policy failures | **0** | **0** |
| full re-run identical | **true** | **true** |
| rows replayed from stored history | 32 | 32 |
| replay problems | **0** | **0** |
| row-only reproduction problems | **0** | **0** |
| results digest | `9efbf4064d74c54d…` | `fcd37044989c0f43…` |

Totals: **256 matches, 142,141 plies, 0 illegal actions, 0 policy failures, 0
replay or reproduction mismatches.** Both modes were re-run in their entirety —
not sampled — and compared row by row. Opponents are the accepted Phase 4 core
ladder (`random_legal`, `basic_heuristic`, `tactical_rule_based`,
`strategic_rule_based`) under the unchanged `color_swap_same_board` pairing.
Win rates are recorded in the data file only as evidence the matches ran; the
network is untrained and they carry no strength information.

**Test suite**

```text
python -m pytest -q
2301 passed, 2 skipped, 0 failed in 94.26s
```

| | Before Phase 6 | After |
|---|---|---|
| passed | 2,155 | 2,301 |
| failed | 0 | 0 |
| skipped | 2 | 2 |

146 tests added: 43 in `tests/model/test_action_frame.py`, 90 in
`tests/model/test_symmetry.py`, and 13 across the retargeted contract and
checkpoint tests. The two skips are the same pre-existing Phase 4 capability
skips; no test was weakened, disabled or made conditional.

**Completion gates**

| Gate | Result |
|---|---|
| `phase_5_accepted` | ✅ |
| `preexisting_suite_green` | ✅ |
| `model_contract_v2_explicit` | ✅ |
| `engine_action_semantics_unchanged` | ✅ |
| `absolute_round_trips_clean` (20,000, 0 mismatches) | ✅ |
| `reverse_round_trips_clean` (20,000, 0 mismatches) | ✅ |
| `action_frame_is_a_bijection` | ✅ |
| `pinned_geometry_correct` | ✅ |
| `legal_list_and_mask_exact` | ✅ |
| `symmetry_regression_passes` | ✅ |
| `v1_and_v2_checkpoints_fail_loudly` | ✅ |
| `hidden_information_zero_mismatch` (≥10,000 trials) | ✅ |
| `positive_controls_succeed` | ✅ |
| `greedy_and_seeded_reproduce` | ✅ |
| `evaluation_regression_clean` | ✅ |
| `full_suite_green` | ✅ |

**16 / 16 true.**

### 1.5 Checkpoint and policy identity

**Checkpoints.** `policy_action_frame` and `engine_action_frame` are now
*required* checkpoint fields. Under v1, `policy_action_frame` was optional and
defaulted to the running build's frame when absent — precisely the silent
reinterpretation this phase removes. The frame fields are validated **before**
every other semantic field and long before any tensor-shape check, because a
frame mismatch is the failure that would otherwise be invisible: the tensors are
the right shape, the weights load, and the network simply plays mirrored moves
for one colour.

The rejection is symmetric and both halves are tested:

- a v1 file will not load here — including the real
  `checkpoints/integration_model_v1.pt`, which is kept in the repository as a
  genuine on-disk rejection fixture;
- a v2 file would not have loaded there — proved through
  `accepted_under_contract_v1()`, a frozen pure-metadata replica of the v1
  acceptance rule (it reads no weights and can load nothing), with a companion
  test showing it still accepts a v1 payload so the check cannot pass vacuously.

A refused load leaves the file byte-identical (digest compared before and
after), and no old checkpoint's metadata is mutated on load.

**Policy identity.** The frame change alters which move the same weights select,
so it took the identity with it:

```text
integration_model_v1_greedy  @ 0.1.0   ->  integration_model_v2_greedy  @ 0.2.0
integration_model_v1_sampled @ 0.1.0   ->  integration_model_v2_sampled @ 0.2.0
```

Phase 5 result rows describe a genuinely different policy and must not be
compared with v2 rows. Neither identity is in the Phase 4 catalogue, so the
accepted Phase 4 ladder, audits and league membership are untouched.

The v2 fixture checkpoint was rebuilt from the same seed (`20250501`) as the
accepted Phase 5 fixture and has a **bit-identical state-dict digest**
(`5b3a8dda…5768`). Only the recorded semantics differ, so every behavioural
difference measured here is attributable to the frame and not to different
weights.

### 1.6 Symmetry regression

The instrument is the mirrored-game pair already accepted in Phase 2
(`tests/observation/test_perspective.py`): a game and its colour-swapped,
180-degree-rotated twin, advanced through mirrored actions. The twin's first
player is blue, which is what makes the ply counts match while the acting
colours are opposite — so both positions are genuinely reachable states produced
by the frozen engine, not hand-built ones.

For each pair, the two equivalent roles must receive identical normalized
observations and identical normalized legal masks, and the same deterministic
network must emit the **same normalized action**, whose absolute forms are the
corresponding rotated moves.

The negative control is what gives this teeth: replaying the retired v1 rule
(identical logits, selected over the engine's absolute identifiers) must
*disagree* with the mirror. It does. Without that control every other assertion
here would also pass on an implementation that never converted anything, because
both frames coincide for red.

### 1.7 Deviations and limitations

1. **`scripts/run_phase5.py` was modified**, which is outside the "files you
   own" list. It loads the fixture checkpoint, so under v2 it could not run at
   all. On the user's explicit decision it now points at
   `checkpoints/integration_model_v2.pt`, its two negative cases were retargeted
   (the values that were "wrong" under v1 are the correct ones under v2, so
   `wrong_model_contract` and `wrong_policy_frame` now carry the retired v1
   values, and a `wrong_engine_frame` case was added), and a `--data-directory`
   option was added so it can be re-run without overwriting accepted evidence.
   No assertion was weakened; every case still asserts the same property.
2. **The accepted Phase 5 artifacts under `reports/phase_5_data/` were not
   overwritten.** They remain the record of the accepted Phase 5 run at commit
   `1d8e7cb`. `run_phase5.py` was instead re-run in full to a scratch directory
   to confirm it still reaches **22 / 22 PASS** under `model_contract_v2`
   (10,000 hidden-information trials, 0 mismatches; 20/20 checkpoint negatives
   rejected; 1,024 gauntlet matches, 584,411 plies, clean). Those numbers are
   evidence that the migration did not break the Phase 5 harness; they are *not*
   a re-acceptance of Phase 5 and do not replace its recorded results.
3. **Phase 5 gauntlet results are not reproducible under v2 and are not meant to
   be.** The policy identity changed with the decision rule (§1.5).
4. **Playing strength is meaningless here.** The network is untrained. Win rates
   appear in the data file only to show that matches ran clean, and no
   architecture or strength claim is drawn from them.
5. **`scripts/run_phase6_agent01.py` imports the mirrored-game helpers from
   `tests/observation/test_perspective.py`.** This follows existing precedent
   (`run_phase2_validation.py` and `run_phase5.py` both import from `tests/`) and
   was preferred over duplicating a subtle construction in two places.
6. **The evaluation regression is a subset**, not the full Phase 5 gauntlet
   size: 256 matches across both policy modes and both colours, chosen as a
   defensible subset within the agreed runtime budget. Determinism is checked by
   re-running every match, not by sampling.

### 1.8 Data files

```text
reports/phase_6_data/agent_01_model_contract_v2.json
```

Every headline number in this section also exists in that file.

### 1.9 Handoff notes for Agent 2

Agent 2 builds the candidate Transformer family C0–C6. **Do not invent another
action frame and do not duplicate the conversion logic.** The public surface to
build against:

**Contract — `stratego.model.contract`**

```python
MODEL_CONTRACT_VERSION   # "model_contract_v2"
TOKEN_SQUARE_FRAME       # "perspective_normalized_squares"
POLICY_ACTION_FRAME      # "perspective_normalized_squares"
ENGINE_ACTION_FRAME      # "absolute_engine_squares"
ACTION_ENCODING_VERSION  # "source_destination_10000_v1"
LEGACY_CONTRACT_V1       # frozen record of what v1 claimed; recognition only
contract_summary()       # serialisable statement of all of the above
```

Shapes every candidate must honour, validated by
`ModelOutputs.validated(...)` / `build_model_outputs(...)`:

```text
input          [B, 127, 10, 10]      tokens  [B, 100, 127]
policy logits  [B, 10000]            normalized source/destination
value logits   [B, 3]                WIN, DRAW, LOSS, acting player
belief logits  [B, 100, 12]          unresolved hidden opponent pieces only
```

Use `validate_policy_logits`, `validate_value_logits`, `validate_belief_logits`
and `validate_observation_batch` rather than writing new shape checks;
`ModelContractError` is the one failure type.

**Tokenization — `stratego.model.tokenization`**

```python
observation_to_tokens(observation)        # [B,127,10,10] -> [B,100,127]
tokens_to_observation(tokens)             # exact inverse
observation_batch_from_numpy(...)         # engine observation(s) -> tensor
tokenize_numpy_observation(...)           # the full engine-to-model input path
```

Token `i` is normalized row-major square `i`. This is a pure relayout — do not
add a positional reordering of your own; positional information belongs in the
model, not in the token order.

**Action frame — `stratego.model.action_frame` (the only converter)**

```python
absolute_action_to_model(action_id, acting_player)
model_action_to_absolute(action_id, acting_player)
absolute_legal_actions_to_model(legal_actions, acting_player)
model_legal_actions_to_absolute(legal_actions, acting_player)
absolute_legal_mask_to_model(mask, acting_player)
model_legal_mask_to_absolute(mask, acting_player)
action_frame_summary()
ActionFrameError    # subclasses ModelContractError
```

Red is the identity; blue maps `square -> 99 - square` on both endpoints. A
candidate model should never see an absolute action id at all.

**Normalized legality.** Convert the engine's products, never rebuild them.
The engine remains the final legality authority, and a model may score an
illegal index arbitrarily — selection happens over the converted legal set.

**Checkpoints — `stratego.model.checkpoint`**

```python
save_checkpoint(model, path, ...)         # writes both frame fields
load_checkpoint(path, device=, dtype=)    # semantic checks before shape checks
validate_checkpoint_payload(payload)
accepted_under_contract_v1(payload)       # frozen v1 rule, recognition only
CheckpointError / CheckpointFormatError / CheckpointCompatibilityError
```

`REQUIRED_FIELDS` now includes `policy_action_frame` and `engine_action_frame`.
A new architecture id must be added to the architecture check rather than
loosening it, and any change to a shape, an ordering or a frame requires a new
`MODEL_CONTRACT_VERSION` — never a silent reinterpretation.

**Policy adapter — `stratego.model.policy_adapter`**

`NeuralCheckpointPolicy` already performs the whole normalized decision path.
A new architecture should be loadable by it without touching `decide`; if it is
not, fix the architecture's interface rather than adding a second decision path.
`greedy_action` / `categorical_action` are pure and frame-agnostic — they take
whatever identifiers they are given, so they must be handed **normalized** ids.

## 2. Agent 2 — Candidate Architecture Family

**Status: PASS** — 15 / 15 completion gates true.

### 2.1 Prerequisite verification

Agent 1 was verified from its real artifact, `reports/phase_6_data/agent_01_model_contract_v2.json`, not assumed:

| Check | Required | Found |
|---|---|---|
| Agent 1 status | `PASS` | `PASS` |
| Agent 1 completion gates | all true | 16 / 16 |
| model contract | `model_contract_v2` | `model_contract_v2` |
| token frame | perspective-normalized | `perspective_normalized_squares` |
| policy frame | perspective-normalized | `perspective_normalized_squares` |
| engine frame | absolute | `absolute_engine_squares` |
| action encoding | unchanged | `source_destination_10000_v1` |

The last five rows are read from the **live** constants in `stratego.model.contract` as
well as from Agent 1's file: the artifact says Agent 1 passed, the constants say this
process is running under the contract that pass was about.

Pre-existing suite, measured **before any Agent 2 edit**:

```text
python -m pytest -q
2301 passed, 2 skipped, 0 failed in 92.25s
```

That is Agent 1's accepted tree (2,155 Phase 5 tests plus Agent 1's 146 additions),
green and unmodified.

### 2.2 What was built

One configurable family, `stratego_transformer_v1`, and seven configurations. There are
no per-candidate model classes and no branches on candidate id anywhere in the network:
C0 and C6 differ by exactly four integers.

```text
stratego/model/architecture_configs.py   the C0-C6 ladder, as frozen serializable configs
stratego/model/production_model.py       the one network those configs describe
stratego/model/base.py                   StrategoModel: the interface both networks implement
```

Everything that is *not* one of the four scaling integers — activation, LayerNorm
epsilon, bias policy, dropout policy, attention implementation, initialization scheme,
head designs — is a module-level family constant, identical for every candidate by
construction and folded into every digest. That is what makes Agent 3's measurements a
comparison of sizes rather than a comparison of architectures.

```text
[B,127,10,10]
 -> [B,100,127] normalized tokens          (pure relayout, row-major)
 -> input projection to width D
 -> + learned row embedding + learned column embedding
 -> `blocks` x pre-norm block: LN -> MHSA -> residual -> LN -> FF -> residual
 -> final LayerNorm                        [B,100,D] shared representation
    ├─ policy  L[i,j] = (Q_i·K_j)/sqrt(D) + b_source[i] + b_dest[j] -> [B,10000]
    ├─ value   mean pool -> Linear -> GELU -> Linear                -> [B,3]
    └─ belief  per-token Linear                                     -> [B,100,12]
```

No causal or padding mask, no candidate-specific observation features, no absolute
Red/Blue input feature, and no separate belief decoder.

**Position representation.** `h[r,c] = W_x x[r,c] + e_row[r] + e_col[c]` — ten row
vectors and ten column vectors, not one hundred per-square vectors. `token_rows` and
`token_columns` are non-persistent buffers, so a checkpoint never carries constants
disguised as weights, and tests pin `token i -> (i // 10, i % 10)` against
`tokenization.square_to_row_column` with distinguishable embeddings, so a transposed
indexing cannot pass.

**The policy head stays in the model frame.** `production_model.py` does not import
`stratego.model.action_frame` at all (asserted by parsing its imports, not by grepping
text), and it never sees an absolute engine action. Two tests pin the flatten: adding to
`policy_source_bias[7]` moves exactly indices 700–799, and adding to
`policy_destination_bias[3]` moves exactly `{100s + 3}` — which is
`action_id = 100 * source + destination` read in normalized squares.

### 2.3 The candidate ladder

Every row is the literal instruction table. Each width divides evenly across its head
count, so **no row violated a hard PyTorch constraint and no adjustment was made**;
`ladder_adjustments` is empty in the data file.

| ID | Width | Blocks | Heads | Feed-forward | Head dim | Trainable parameters | fp32 | fp16 | Checkpoint | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | 64 | 2 | 4 | 256 | 16 | 123,223 | 492,892 B | 246,446 B | 504,965 B | small control |
| C1 | 128 | 4 | 4 | 512 | 32 | 863,959 | 3,455,836 B | 1,727,918 B | 3,473,613 B | small practical |
| C2 | 192 | 4 | 6 | 768 | 32 | 1,922,519 | 7,690,076 B | 3,845,038 B | 7,707,853 B | wider |
| C3 | 192 | 6 | 6 | 768 | 32 | 2,812,247 | 11,248,988 B | 5,624,494 B | 11,272,469 B | deeper |
| C4 | 256 | 6 | 8 | 1,024 | 32 | 4,978,391 | 19,913,564 B | 9,956,782 B | 19,937,173 B | medium-large |
| C5 | 256 | 8 | 8 | 1,024 | 32 | 6,557,911 | 26,231,644 B | 13,115,822 B | 26,261,035 B | deeper medium-large |
| C6 | 384 | 8 | 8 | 1,536 | 48 | 14,702,807 | 58,811,228 B | 29,405,614 B | 58,840,619 B | paper-width/depth ceiling reference |

C6 is an upper-region benchmark reference, not a presumed choice. Random-weight playing
strength was not measured and must not influence selection.

Per-component accounting (trainable parameters; the four groups sum to the total, which
a test asserts rather than assumes):

| ID | Encoder | Policy head | Value head | Belief head |
|---|---:|---:|---:|---:|
| C0 | 109,568 | 8,520 | 4,355 | 780 |
| C1 | 812,288 | 33,224 | 16,899 | 1,548 |
| C2 | 1,808,256 | 74,312 | 37,635 | 2,316 |
| C3 | 2,697,984 | 74,312 | 37,635 | 2,316 |
| C4 | 4,776,960 | 131,784 | 66,563 | 3,084 |
| C5 | 6,356,480 | 131,784 | 66,563 | 3,084 |
| C6 | 14,253,312 | 295,880 | 148,995 | 4,620 |

### 2.4 Architecture identity and checkpoint compatibility

`checkpoint.py` previously named `IntegrationModel` directly. Rather than loosening its
checks to accept "any module", the architectures are now **registered**:

```python
registered_architectures()  # ('integration_model_v1', 'stratego_transformer_v1')
```

An unregistered `model_architecture_id` is refused exactly as a wrong rules version is,
and re-registering an id raises rather than silently overwriting — an import-order
dependent change of meaning is the failure class this module exists to prevent. Every
Phase 5 checkpoint test passes unchanged.

**Why configuration equality, not shape compatibility, is the gate.**
`nn.MultiheadAttention` packs all heads into one `(3D, D)` projection, so head count
never appears in a parameter shape: C2 (192/4/**6**/768) and a 192/4/**4**/768 variant
have byte-identical state dicts and `load_state_dict` would succeed silently. The
harness confirms that shape identity first (`shape_compatible_pair_confirmed: true`) and
then shows the load being refused anyway. Seven negative cases, 7/7 refused:

| Case | Result |
|---|---|
| C2 weights into a C3 model | rejected |
| identical shapes, different head count | rejected |
| payload claiming `C3` while carrying C2's shape | rejected |
| `integration_model_v1` weights into a candidate | rejected |
| candidate weights into `integration_model_v1` | rejected |
| unregistered architecture id (`ataraxos_full_v1`) | rejected |
| `expected_configuration` naming a different candidate | rejected |

`load_checkpoint(..., expected_architecture_id=, expected_configuration=)` makes the
identity check available when the caller knows what it asked for;
`load_checkpoint_into(model, path)` makes it **mandatory**, because a target exists.

`integration_model_v1` is untouched as the Phase 5 fixture: same weights, same shapes,
same architecture id, now also declaring the shared `StrategoModel` interface.

### 2.5 Determinism

For every candidate, all true:

| Property | Result |
|---|---|
| same seed → bit-identical CPU state dict | 7 / 7 |
| same seed → identical state-dict digest | 7 / 7 |
| different seed → different weights | 7 / 7 |
| configuration round-trips through JSON | 7 / 7 |
| rebuilt from `(config, seed)` alone → identical weights | 7 / 7 |
| exact parameter count reproducible | 7 / 7 |
| configuration digest stable | 7 / 7 |
| evaluation-mode CPU forward deterministic | 7 / 7 |

One declared seed for the whole benchmark family: **`FAMILY_INITIALIZATION_SEED =
20250601`**. Initialization draws from an explicit `torch.Generator`, never the global
RNG, so reproducibility is a property of the model rather than of the caller.

Architecture family digest: `5b57dd3a0c1ae6fd…` (full value in the data file). It covers
the family constants *and* every candidate digest, so a changed activation or a changed
row is visible as a changed identity rather than as a silent difference between runs.

### 2.6 Smoke validation

Batch 4, CPU forward passes fed with **real engine positions** (seeded random games at
plies 0, 15, 46 and 91, both colours acting) rather than noise:

| Check | Result |
|---|---|
| CPU construction + forward, exact shapes, finite | 7 / 7 |
| backward connectivity (every parameter: gradient present, finite, non-zero) | 7 / 7 |
| MPS float32 construction + forward, finite | 7 / 7 |
| MPS float16 construction + forward, finite | 7 / 7 |
| checkpoint save/load round-trip under `model_contract_v2` | 7 / 7 |
| reloaded weights and outputs identical to the original | 7 / 7 |

MPS was genuinely used — `mps_built: true`, `mps_available: true`, results reported from
`mps:0` with `torch.mps.synchronize()` — and there is no CPU fallback path in the MPS
check: a failure would have been recorded as `ok: false` with its exception text. No
candidate failed for any reason, size-related or otherwise. A test also pins CPU/Metal
agreement to `1e-4` for C0, which is a numerical check, not a benchmark.

Backward passes here exist only to measure gradient connectivity, as Phase 6 permits.
Nothing was trained, no optimizer was constructed, and no hyperparameter was tuned.

The forward timings in `cpu_forward_results.*.smoke_seconds` are single unwarmed small
batches recorded as evidence that the pass ran. **They are not benchmark numbers** and
must not be read as performance evidence; Agent 3 owns that.

### 2.7 Files created and modified

Created:

```text
stratego/model/architecture_configs.py
stratego/model/base.py
stratego/model/production_model.py
tests/model/test_architecture_family.py
scripts/run_phase6_agent02.py
reports/phase_6_data/agent_02_architecture_family.json
```

Modified:

```text
stratego/model/checkpoint.py       architecture registry; identity checks on load
stratego/model/integration_model.py  now declares the shared StrategoModel interface
stratego/model/policy_adapter.py   accepts any StrategoModel, not one named class
stratego/model/__init__.py         exports the new public surface
reports/phase_6_implementation_report.md
```

`stratego/engine/` was not touched. Neither were the rules, the observation, the action
encoding, Phase 4 evaluation semantics, or `stratego_project_docs/`.

### 2.8 Tests and commands

```text
python -m pytest -q                                        # before: 2301 passed, 2 skipped
python -m pytest tests/model/test_architecture_family.py -q # 82 passed
python scripts/run_phase6_agent02.py                       # PASS, 15/15 gates, 98.06s
python -m pytest -q                                        # after: 2383 passed, 2 skipped
```

82 new tests, no test removed, weakened or skipped. The two skips are the same
pre-existing Phase 4 capability skips Agent 1 recorded. The five Metal tests execute on
this host and skip only where Metal is unavailable; the hard MPS requirements are
enforced by the harness, which never substitutes CPU.

### 2.9 Completion gates

| Gate | Result |
|---|---|
| Agent 1 PASS verified | ✅ |
| pre-existing suite green | ✅ 2301 passed, 0 failed |
| one configurable family implements all candidates | ✅ |
| C0–C6 configs explicit and serializable | ✅ |
| construction is deterministic | ✅ |
| exact parameter counts recorded | ✅ |
| policy/value/belief outputs match `model_contract_v2` | ✅ |
| policy logits remain perspective-normalized | ✅ |
| no privileged inputs exist | ✅ |
| CPU smoke checks pass for all candidates | ✅ 7 / 7 |
| MPS float32 smoke passes for every candidate | ✅ 7 / 7 |
| MPS float16 honestly attempted | ✅ 7 / 7 attempted, 7 / 7 passed |
| backward connectivity passes | ✅ 7 / 7 |
| checkpoint/config mismatch rejection works | ✅ 7 / 7 |
| full suite green | ✅ 2383 passed, 2 skipped, 0 failed |

### 2.10 Deviations and limitations

1. **`stratego/model/base.py` is beyond the suggested file list.** The instruction's file
   list is explicitly a suggestion; supporting two architectures required either a shared
   interface or a loosened `isinstance` check in `checkpoint.py` and `policy_adapter.py`,
   and a written-down interface was preferred to a weakened check.
2. **`checkpoint.py` and `policy_adapter.py` were modified**, which are Phase 5 files.
   Both were hard-bound to `IntegrationModel`, so no candidate could be saved, loaded or
   played without touching them. The change is additive — a registry plus optional
   identity arguments — and every pre-existing checkpoint and adapter test passes
   unchanged.
3. **`model_architecture_id` is one id for the whole family**, with the candidate carried
   in the configuration, rather than seven registered ids. Seven ids would have made the
   registry a hand-maintained list of the exact kind the instruction forbids; identity is
   enforced instead by configuration equality, which is strictly stronger — it also
   catches an off-ladder configuration that no id could describe.
4. **Dropout is a configuration field defaulting to `0.0`** for every candidate, and the
   dropout modules exist unconditionally so that a checkpoint round-trips across dropout
   values. Two tests prove evaluation mode is deterministic at `dropout=0.5` while train
   mode is not, so a later agent turning dropout on cannot silently make benchmarks
   non-deterministic.
5. **Belief supervision semantics are unchanged** from Phase 5 (hidden opponent pieces
   only, `ignore_index = -100`), and the existing `multi_head_loss` is exercised against
   a candidate unmodified. No separate belief decoder was built.
6. **The smoke batch is 4 positions.** Deliberately small: this is a correctness harness,
   and a larger sweep would invite its timings being read as a benchmark. Agent 3 owns
   batch sweeps.
7. **Candidate checkpoints are written to a temporary directory** and deleted after their
   size and digest are recorded. Seven initialized candidates are ~127 MB of weights that
   carry no information `(config, seed)` does not already carry.

### 2.11 Data files

```text
reports/phase_6_data/agent_02_architecture_family.json
```

Every headline number in this section also exists in that file.

### 2.12 Handoff notes for Agent 3

Benchmark these exact candidates. **No architecture edits**: if a candidate needs a
change to be benchmarkable, that is a finding to report, not a patch to apply.

**Candidate table** — `stratego.model.architecture_configs`

```python
CANDIDATE_IDS        # ('C0','C1','C2','C3','C4','C5','C6'), smallest first
CANDIDATES[id]       # the frozen CandidateConfig
CANDIDATE_ROLES[id]  # the instruction's role text
candidate_table()    # report-ready rows, digests included
candidate_configs()  # serialized configs, keyed by id
config_digests()     # id -> full configuration digest
family_summary()     # the whole family in one serializable dict
architecture_family_digest()
ARCHITECTURE_FAMILY          # 'stratego_transformer_v1'
ARCHITECTURE_FAMILY_VERSION  # 'architecture_family_v1'
FAMILY_INITIALIZATION_SEED   # 20250601 -- use this unless a sensitivity check needs otherwise
```

**Model construction** — `stratego.model.production_model`

```python
build_candidate_model('C3', seed=FAMILY_INITIALIZATION_SEED, device='mps', dtype=torch.float16)
build_all_candidates(seed=...)          # every candidate, CPU float32
ProductionModel(config_or_candidate_id, seed=...)
model.parameter_count() / trainable_parameter_count() / parameter_breakdown()
model.parameter_bytes(torch.float16)
model.architecture_summary()            # config, digest, counts, seed, provenance note
```

Models are always built on CPU and then moved, so CPU and Metal start from bit-identical
float32 weights and any difference measured later is the kernels, not initialization.

**Benchmark input and output validation**

```python
benchmark_observation_batch(batch, seed=, device=, dtype=)  # [B,127,10,10], contract-valid
benchmark_token_batch(batch, seed=, device=, dtype=)        # [B,100,127]
validate_candidate_outputs(outputs, batch=, require_finite=True)
```

These are *shape*-valid deterministic inputs, which is what throughput measurement wants.
Anything that depends on the values — legality, decisions, hidden information — must use
real engine positions: `stratego.training.mps_benchmark.build_position_pool(...)`, which
returns `(P, 100, 127)` tokens with dense masks and legal lists.

**Checkpoints**

```python
save_checkpoint(model, path)
load_checkpoint(path, device=, dtype=, expected_architecture_id=, expected_configuration=)
load_checkpoint_into(model, path)     # identity check is mandatory here
registered_architectures()
```

**MPS limitations found: none.** All seven candidates construct and run on Metal in both
float32 and float16 on this host (Apple M4 Pro, 48 GB, torch 2.13.0, Python 3.13.2), with
finite outputs and exact contract shapes at batch 4. C6 at 14.7 M parameters is 56.09 MiB
of float32 weights, so capacity limits — if any exist — will come from activations and
batch size, which is Agent 3's measurement to make.

**Do not** treat any timing in `agent_02_architecture_family.json` as a benchmark, and do
not use random-weight playing strength as evidence for anything.

## 3. Agent 3 — Standalone MPS and Training-Step Benchmark

**Status: PASS** — 16 / 16 completion gates true.

### 3.1 Prerequisite verification

Agents 1 and 2 were verified from their real artifacts *and* from the live build,
because the two claims are different: an artifact says an agent passed, the
constants say this process is running under the contract that pass was about.

| Check | Required | Found |
|---|---|---|
| Agent 1 status / gates | `PASS`, all true | `PASS`, 16 / 16 |
| Agent 2 status / gates | `PASS`, all true | `PASS`, 15 / 15 |
| live `MODEL_CONTRACT_VERSION` | `model_contract_v2` | `model_contract_v2` |
| live token / policy frame | perspective-normalized | `perspective_normalized_squares` |
| live engine frame | absolute | `absolute_engine_squares` |
| live action encoding | unchanged | `source_destination_10000_v1` |
| architecture family digest | matches Agent 2's record | `5b57dd3a0c1ae6fd…` ✅ |

**Every candidate was rebuilt and required to reproduce before any timing.** Each
of C0–C6 was reconstructed from its stored configuration and its configuration
digest and exact trainable parameter count compared against
`agent_02_architecture_family.json`:

| | C0 | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trainable parameters | 123,223 | 863,959 | 1,922,519 | 2,812,247 | 4,978,391 | 6,557,911 | 14,702,807 |
| digest reproduces | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

7 / 7 digests and 7 / 7 parameter counts match exactly. This is the check that
would fail if an architecture had been edited to improve its benchmark results,
which the instructions forbid and which did not happen.

Pre-existing suite, measured **before any Agent 3 edit**:

```text
python -m pytest -q
2383 passed, 2 skipped, 0 failed in 92.75s
```

That is Agent 2's accepted tree, green and unmodified.

### 3.2 What was built

```text
stratego/model/benchmark_helpers.py   corpus, timing, numerical checks, classification
scripts/run_phase6_agent03.py         the acceptance harness
tests/model/test_phase6_benchmarks.py 62 tests of the instrument itself
```

Phase 3/4's accepted `stratego/training/mps_benchmark.py` is imported **read-only**
for device detection, Metal counters and synchronisation. It was not modified,
and neither was the engine, the Phase 4 evaluation, or any Agent 1–2 file.

**Fairness is structural.** Every candidate goes through the same functions with
the same corpus rows, warmup, repetition policy, synchronisation, loss
definitions and target tensors. There is no per-candidate branch anywhere in the
module, so there is no place for a favoured candidate to be treated differently.

**A label is a measurement, not an intention.** Every row records the device and
dtype read back off the *output tensor*, and `verify_execution_labels` raises
rather than records on a mismatch — the one failure the harness refuses to
downgrade into a row, because a row that lies about where it ran would silently
contaminate every comparison downstream.

### 3.3 The input corpus

4,096 real acting-player positions collected from the frozen engine through the
public `BatchSimulator` surface, labelled with everything the benchmark needs.

| Property | Value |
|---|---|
| positions | 4,096 |
| digest | `563a7ccd941e996b6f65e5a6980f57b0065447f7856d03a04cda9b88d6d434cc` |
| rebuilt-from-recipe digest identical | **true** |
| acting colour | red 2,069 / blue 2,027 |
| ply range | 0 – 465 (mean 188.4) |
| legal actions per position | min 3, mean 24.1, max 48 |
| belief-supervised squares | min 8, mean 28.7, max 40 |
| policy targets illegal in the normalized list | **0** |
| policy targets illegal in the normalized mask | **0** |
| normalized list vs normalized mask mismatches | **0 / 4,096** |
| normalized → absolute set mismatches | **0 / 4,096** |
| build time | 1.69 s |

**A defect found and fixed during construction.** The first corpus came back
2,069 red / **2 blue**. The acting player is the ply's parity and every slot in a
sampling round sits at the same ply, so an *even* collection stride samples
nothing but even plies and yields a corpus in which red always acts. That corpus
looks entirely healthy on every other measure and would never once have exercised
the blue branch of the perspective transform — the whole subject of
`model_contract_v2`. The stride is now odd (15), and `build_benchmark_corpus`
**raises** rather than returning a single-colour corpus. Both the fix and the
guard are covered by regression tests.

**Targets, and how they are generated.** These exist only so the backward pass has
something real to differentiate. Nothing is learned from them; no optimizer
exists.

| Head | Source |
|---|---|
| policy | one action drawn per position from that position's own **normalized** legal list with a seeded generator (`TARGET_SEED = 20260812`), then proved legal independently against both the normalized list and the normalized dense mask |
| value | seeded uniform draw over WIN / DRAW / LOSS; carries no game outcome |
| belief | `dense_belief_target` on the real `GameState` for the acting player — unresolved hidden opponent pieces only, normalized squares, `-100` elsewhere |

Belief targets are privileged and are used **only** as a backward-pass target.
They are never a model input; the observation tensor a candidate receives is
`[B, 127, 10, 10]` and nothing is concatenated onto it.

### 3.4 Method

| Setting | Value |
|---|---|
| warmup | 5 iterations, unmeasured |
| measurement | 10–60 iterations, time-boxed at ~1.0 s per point |
| synchronisation | `torch.mps.synchronize()` immediately **before and after** every timed region, and between training-step stages |
| dtype convention | parameters and activations both cast to the row's precision |
| initialisation | `FAMILY_INITIALIZATION_SEED = 20250601`, built on CPU then moved |
| memory APIs | `current_allocated_memory`, `driver_allocated_memory`, `recommended_max_memory`, `ru_maxrss` |

The leading synchronise stops the previous sample's unfinished work being charged
to this one; the trailing one is what makes the number a latency at all, since
MPS dispatch is asynchronous and an unsynchronised timer measures queue
submission. Two tests pin this: one asserts a trailing drain after
`timed_samples` costs under 25% of a single sample, the other that the samples
account for at least 70% of the call's wall-clock time.

**Timing boundaries.** Recorded per row, because a latency without its boundary is
not actionable.

| Boundary | Contains |
|---|---|
| **A** | model forward only; tokens already device-resident, contiguous, in the target dtype |
| **B** | host NumPy observation → copy + contract validation → token relayout → host-to-device transfer → forward |
| **C** | B, plus the normalized legality mask transfer, masked greedy selection in the normalized frame on device, readback, and inverse conversion to absolute engine actions |

### 3.5 Measured results

Environment: macOS 26.5.2 arm64, Python 3.13.2, torch 2.13.0, MPS built and
available, 10 CPU threads, Apple M4 Pro 48 GB (Metal recommended maximum
40.2 GB). Total harness time **1,405.7 s**, of which inference 1,154.1 s,
training 137.1 s, numerical 3.2 s, corpus 3.4 s.

**297 inference rows and 56 training rows. Zero out-of-memory. Zero errors. Zero
non-finite outputs.** Every point in the required matrix — 7 candidates × 7
batch sizes × 2 precisions × 3 boundaries = 294, plus 3 extended rows —
completed successfully.

**Inference throughput, boundary A, positions/s at the best stable batch**

| ID | Parameters | float32 | @ batch | float16 | @ batch | fp16 gain |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 123,223 | **25,363** | 2048 | **27,156** | 2048 | +7% |
| C1 | 863,959 | **12,304** | 2048 | **14,919** | 2048 | +21% |
| C2 | 1,922,519 | **6,785** | 2048 | **8,636** | 3072 | +27% |
| C3 | 2,812,247 | **4,812** | 2048 | **6,071** | 2048 | +26% |
| C4 | 4,978,391 | **3,291** | 2048 | **3,968** | 2048 | +21% |
| C5 | 6,557,911 | **2,567** | 2048 | **3,016** | 2048 | +17% |
| C6 | 14,702,807 | **1,154** | 2048 | **1,293** | 2048 | +12% |

Throughput is effectively flat from batch 512 upward: across all fourteen
candidate/precision curves, batch 512 already reaches 93.8–100% of that curve's
peak. The knee is at batch 256–512, and beyond it the M4 Pro is saturated. Batch 1 is latency-bound
(0.96–4.39 ms, 228–1,046 positions/s across all candidates and both precisions),
which is a statement about dispatch overhead rather than about the networks.

**Preprocessing and action-selection overhead, batch 2048, float32**

| ID | A (ms) | B (ms) | B − A | C (ms) | C − B |
|---|---:|---:|---:|---:|---:|
| C0 | 80.7 | 98.1 | +17.3 (+21%) | 95.1 | −2.9 |
| C1 | 166.4 | 176.2 | +9.7 (+6%) | 178.8 | +2.7 |
| C2 | 308.0 | 320.2 | +12.2 (+4%) | 318.1 | −2.2 |
| C3 | 428.2 | 442.6 | +14.4 (+3%) | 446.2 | +3.7 |
| C4 | 633.6 | 654.3 | +20.7 (+3%) | 658.5 | +4.2 |
| C5 | 840.6 | 856.9 | +16.4 (+2%) | 861.9 | +5.0 |
| C6 | 1,866.4 | 1,878.7 | +12.3 (+1%) | 1,882.3 | +3.6 |

Two findings. **Preprocessing is a roughly constant ~10–20 ms at batch 2048**,
independent of candidate size — it is the observation copy, contract validation
and relayout, which do not depend on the network. It therefore costs C0 21% and
C6 1%: the smaller the model, the more the input path dominates, and a small
candidate's advantage is partly eaten before the model runs.

**The normalized legality and action-selection path is effectively free.** C − B
is within run-to-run noise at every size, and is *negative* in two rows — which
means the honest reading is an upper bound of a few milliseconds rather than a
positive measured cost. Selecting in the normalized frame and converting back to
absolute engine actions does not meaningfully change the inference budget.

**Action legality.** Across 99 selection-validity reports covering **79,246
selections**: **0** illegal in the normalized frame and **0** illegal after
conversion to absolute engine actions.

**Numerical checks — CPU float32 reference vs MPS**

Tolerances were declared before measuring and applied identically to every
candidate regardless of depth (nothing was loosened for a deeper model):

```text
MPS float32:  policy/belief logits max|e| <= 1e-4    value probabilities <= 1e-5
MPS float16:  policy/belief logits max|e| <= 5e-2    value probabilities <= 5e-3
```

| | policy max\|e\| | belief max\|e\| | value prob max\|e\| | crafted-margin | natural | illegal |
|---|---:|---:|---:|---:|---:|---:|
| C0 f32 | 7.30e-07 | 8.05e-07 | 8.94e-08 | 256/256 | 256/256 | 0 |
| C1 f32 | 1.48e-06 | 1.61e-06 | 8.94e-08 | 256/256 | 256/256 | 0 |
| C2 f32 | 1.85e-06 | 1.61e-06 | 8.94e-08 | 256/256 | 256/256 | 0 |
| C3 f32 | 1.62e-06 | 1.79e-06 | 5.96e-08 | 256/256 | 256/256 | 0 |
| C4 f32 | 2.09e-06 | 2.27e-06 | 1.19e-07 | 256/256 | 256/256 | 0 |
| C5 f32 | 1.88e-06 | 1.55e-06 | 8.94e-08 | 256/256 | 256/256 | 0 |
| C6 f32 | 1.79e-06 | 1.79e-06 | 1.49e-07 | 256/256 | 256/256 | 0 |
| C0 f16 | 1.46e-03 | 1.71e-03 | 1.27e-04 | 256/256 | 255/256 | 0 |
| C1 f16 | 1.99e-03 | 3.22e-03 | 1.34e-04 | 256/256 | 254/256 | 0 |
| C2 f16 | 1.93e-03 | 2.26e-03 | 9.19e-05 | 256/256 | 256/256 | 0 |
| C3 f16 | 2.36e-03 | 2.65e-03 | 8.50e-05 | 256/256 | 252/256 | 0 |
| C4 f16 | 2.30e-03 | 2.71e-03 | 8.60e-05 | 256/256 | 255/256 | 0 |
| C5 f16 | 2.05e-03 | 2.49e-03 | 1.14e-04 | 256/256 | 251/256 | 0 |
| C6 f16 | 2.28e-03 | 2.53e-03 | 1.76e-04 | 256/256 | 256/256 | 0 |

**14 / 14 comparisons pass, all outputs finite.** The worst float32 error is
2.09e-06 against a 1e-4 tolerance (48× headroom) and the worst float16 error is
3.22e-03 against 5e-2 (15× headroom), with no depth trend — C6 at 8 blocks is no
worse than C0 at 2.

**Crafted-margin action agreement is exact everywhere: 14 × 256 / 256.** A fixed
+5.0 bonus is added to one designated legal action per position on both sides, so
the intended choice wins by more than any rounding can move it; the harness also
verifies the margin actually dominates on the reference itself, so the check
cannot pass vacuously. No crafted-margin flip occurred anywhere and none was
ignored.

Natural-corpus agreement in float16 is 251–256 / 256. Those 0–5 flips per
candidate are genuine near-ties in untouched logits, reported without being
judged — they carry no strength meaning on random weights.

**Relative error is reported honestly and not used as a pass criterion.**
Unfiltered maximum relative error reaches **9.37e+03** (C0 float16 policy) and
1.06e+03 (C0 float16 belief). Those ratios describe near-zero denominators, not
the network: the same entries have absolute errors of 1.46e-03 and 1.71e-03.
Relative error is therefore also reported restricted to entries whose reference
magnitude exceeds a predeclared floor of 1e-3, where it lands at ~1.1 for
float16 and ~5e-4 for float32. Both figures are in the data file.

**Training-step benchmark — batch 256, the largest required**

One step is forward + policy loss + W/D/L value loss + masked hidden-only belief
loss + backward. **No optimizer, no parameter update.**

| ID | prec | forward | loss | backward | total | examples/s | finite loss | finite grads |
|---|---|---:|---:|---:|---:|---:|---|---|
| C0 | f32 | 10.2 | 2.8 | 15.4 | 28.5 | **8,977** | ✅ | ✅ |
| C1 | f32 | 28.8 | 3.4 | 54.0 | 86.3 | **2,966** | ✅ | ✅ |
| C2 | f32 | 47.1 | 3.8 | 90.5 | 142.0 | **1,803** | ✅ | ✅ |
| C3 | f32 | 68.3 | 3.8 | 132.6 | 204.8 | **1,250** | ✅ | ✅ |
| C4 | f32 | 99.2 | 3.8 | 195.9 | 298.9 | **856** | ✅ | ✅ |
| C5 | f32 | 131.5 | 4.1 | 259.8 | 395.3 | **648** | ✅ | ✅ |
| C6 | f32 | 232.4 | 4.5 | 475.7 | 712.9 | **359** | ✅ | ✅ |
| C6 | f16 | 234.7 | 4.2 | 403.3 | 642.2 | **399** | ✅ | ✅ |

All 56 training rows completed. Backward costs 1.9–2.05× forward for C1–C6 (1.51×
for C0, which is small enough that dispatch overhead still shows); the three
losses together cost 2.8–4.5 ms regardless of candidate size, because they are
dominated by the 10,000-wide masked `log_softmax` rather than by the network.

**Pure float16 backward works on every candidate.** No autocast, no loss scaling,
no optimizer — the honest half-precision path — and losses and gradients were
finite for all 7 candidates at all 4 batch sizes. Total loss agrees with float32
to 4–6 decimal places (e.g. C0 5.941586 vs 5.941531), and shared-encoder gradient
norms agree to within 0.01–0.80%. The speed benefit is modest — 1.2–4.3% at batch
256 for C0–C5, and 11.0% for C6 — because the accepted loss functions upcast to
float32 internally, identically for every candidate.

Every one of the four parameter groups received a non-zero, finite gradient in
every row: shared encoder, policy head, value head and belief head are all
genuinely connected.

**Memory — the frontier is compute-bound, not memory-bound**

| ID | peak Metal driver | fraction of 40.2 GB recommended max |
|---|---:|---:|
| C0 | 3.35 GB | 8.3% |
| C1 | 3.28 GB | 8.2% |
| C2 | 4.51 GB | 11.2% |
| C3 | 4.51 GB | 11.2% |
| C4 | 5.13 GB | 12.8% |
| C5 | 5.61 GB | 13.9% |
| C6 | **9.30 GB** | **23.1%** |

**Nothing came close to the memory ceiling.** Even C6 at 14.7 M parameters and
batch 2048 used under a quarter of the recommended maximum. No candidate ran out
of memory, and none was skipped by the memory guard. Per the instruction, the
host was never deliberately driven into OOM or swap: batches above 2,048 were
attempted only where throughput was still improving by ≥2% and Metal use was
below 60% of the recommended maximum. That fired for exactly one configuration —
**C2 float16 at batch 3,072** (8,636 positions/s, a 3% gain over 2,048) — and
every other candidate had already flattened by 2,048. The practical ceiling for
these architectures on this host is therefore **compute, not capacity**, and the
maximum stable batch is reported as "≥ 2,048, not driven to failure" rather than
as a measured OOM point.

**Why C6 is or is not practical.** C6 was measured to the same depth as every
other candidate: it runs, it is numerically clean in both precisions, it is
finite under backward, and it uses only 23% of memory. It is impractical purely
on speed — 1,154 positions/s float32 inference and 359 training examples/s, 22×
and 25× slower than C0. That is a compute verdict backed by a complete set of
successful measurements, not an inference from a failure.

### 3.6 Classification

The rule was declared in `benchmark_helpers.py` before any measurement existed,
is evaluated in a fixed order, and reads **only** a projected allowlist of ten
numeric fields:

1. **IMPRACTICAL** if any of: max stable inference batch < 256; max stable
   training batch < 32; the CPU/MPS float32 numerical check fails; best stable
   float32 throughput < 5,000 positions/s; peak Metal use > 80% of the
   recommended maximum.
2. **DOMINATED** if another practical candidate has ≥ parameters **and** ≥ float32
   inference throughput **and** ≥ training throughput **and** ≥ max stable batch,
   with at least one strict improvement.
3. **ADVANCE** otherwise.

The 5,000 positions/s floor is derived, not chosen for convenience: Phase 3
measured a simulation-only numerator of ~96,963 positions/s and an integrated
rate of ~12,838 positions/s. Under serial composition
(`1/integrated = 1/simulation + 1/model`) a model below 5,000 positions/s caps
the integrated pipeline under ~4,755 positions/s before any recording cost.

| ID | Parameters | f32 pos/s | f16 pos/s | train ex/s | max inf batch | max train batch | memory | Class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | 123,223 | 25,363 | 27,156 | 8,977 | ≥2048 | 256 | 8.3% | **ADVANCE** |
| C1 | 863,959 | 12,304 | 14,919 | 2,966 | ≥2048 | 256 | 8.2% | **ADVANCE** |
| C2 | 1,922,519 | 6,785 | 8,636 | 1,803 | ≥3072 | 256 | 11.2% | **ADVANCE** |
| C3 | 2,812,247 | 4,812 | 6,071 | 1,250 | ≥2048 | 256 | 11.2% | IMPRACTICAL |
| C4 | 4,978,391 | 3,291 | 3,968 | 856 | ≥2048 | 256 | 12.8% | IMPRACTICAL |
| C5 | 6,557,911 | 2,567 | 3,016 | 648 | ≥2048 | 256 | 13.9% | IMPRACTICAL |
| C6 | 14,702,807 | 1,154 | 1,293 | 359 | ≥2048 | 256 | 23.1% | IMPRACTICAL |

Determinism was measured, not assumed: classifying the same summaries again, and
in reverse order, produced identical verdicts. Strength-independence was
demonstrated with a positive control — injecting `win_rate`, `elo`,
`gauntlet_score` and `match_results` fields that would reverse the ordering
changed nothing, because `classification_inputs` projects the summary onto ten
declared keys and a strength-shaped key added to that list raises.

**Two things about this classification the reader should not have to dig for.**

**C3 is a near miss, and the margin is 3.8%.** C3 reached 4,812 positions/s
float32 against a floor of 5,000 — it fails by less than four percent. Its
float16 path reaches **6,071 positions/s**, which clears the floor by 21%, and
its float16 numerical check passes with the same headroom as every other
candidate. The floor was declared against float32 before the numbers existed and
has been applied as declared rather than moved after seeing where C3 landed, but
a 3.8% miss on one precision is not a robust verdict. C3 is the obvious
first candidate for Agent 4 to reinstate, and its float16 measurement is exactly
the kind of specific, measured integration reason the handoff rules require.

**The DOMINATED class is empty, and that is a finding rather than an oversight.**
Domination requires a candidate that is both larger and faster. Across C0–C6
throughput falls monotonically as parameters rise, so no such pair exists and the
domination clause never fired. The practical consequence is that the shortlist
was determined entirely by the throughput floor: there is no redundant candidate
in the ladder, and the choice facing Agents 4 and 6 is a pure capacity-versus-
throughput trade rather than a matter of eliminating waste.

### 3.7 Files created and modified

Created:

```text
stratego/model/benchmark_helpers.py
scripts/run_phase6_agent03.py
tests/model/test_phase6_benchmarks.py
reports/phase_6_data/agent_03_inference_benchmark.csv
reports/phase_6_data/agent_03_training_step_benchmark.csv
reports/phase_6_data/agent_03_architecture_shortlist.json
```

Modified:

```text
reports/phase_6_implementation_report.md
```

**No other file in the repository was modified.** `stratego/engine/`,
`stratego/training/`, `stratego/evaluation/`, every Agent 1 and Agent 2 module,
the Phase 4 evaluation semantics, and `stratego_project_docs/` are all untouched.
No architecture was edited.

### 3.8 Tests and commands

```text
python -m pytest -q                                       # before: 2383 passed, 2 skipped
python -m pytest tests/model/test_phase6_benchmarks.py -q  # 62 passed
python scripts/run_phase6_agent03.py                       # PASS, 16/16 gates, 1405.73s
python -m pytest -q                                        # after: 2445 passed, 2 skipped, 0 failed
```

| | Before Agent 3 | After |
|---|---|---|
| passed | 2,383 | **2,445** |
| failed | 0 | **0** |
| skipped | 2 | 2 |

62 tests added, none removed, weakened, disabled or made conditional. The two
skips are the same pre-existing Phase 4 capability skips Agents 1 and 2 recorded.

The tests are about the *instrument*, since a benchmark that ran on CPU, or in the
wrong dtype, or that dropped its failures, would still produce a complete and
plausible report. They prove: configurations reproduce Agent 2's digests and
parameter counts; the corpus is deterministic and its digest covers the targets
as well as the positions; a single-colour corpus is refused; policy targets are
legal in the normalized frame; the timed region leaves no queued Metal work and
the samples account for the wall clock; CPU results cannot be labelled MPS and
float32 cannot be labelled float16 (both barriers, with PyTorch's own device
check removed for the second); OOM and error rows become rows and survive the CSV
writer; unavailable memory renders as `unavailable` and not zero; classification
is deterministic and order-independent; a strength field cannot change a verdict
and a strength-shaped input key raises; and no parameter is updated by a
benchmark training step, checked by comparing weights rather than by trusting a
flag.

### 3.9 Completion gates

| Gate | Result |
|---|---|
| Agents 1–2 PASS verified | ✅ |
| candidate configs and parameter counts reproduce | ✅ 7 / 7 |
| MPS actually used | ✅ every `ok` row observed on `mps:0` |
| deterministic valid corpus recorded | ✅ digest reproduces, 0 legality mismatches |
| fair inference matrix attempted | ✅ 297 / 297 |
| OOM and error rows retained | ✅ (0 occurred; retention proven by test) |
| CPU/MPS numerical checks completed | ✅ 7 / 7 candidates, 14 / 14 comparisons |
| float16 honestly tested | ✅ dtype verified on the output tensor |
| crafted-margin action agreement passes for advancing | ✅ 256 / 256 |
| training-step benchmark completed | ✅ 56 / 56 |
| losses and gradients finite for advancing | ✅ |
| memory measured and reported | ✅ |
| deterministic classification produced | ✅ order-independent |
| at least three advance when three are viable | ✅ 3 advance |
| no strength-based selection | ✅ positive control passes |
| full suite green | ✅ 2,445 passed, 0 failed |

**16 / 16 true.**

### 3.10 Deviations and limitations

1. **The corpus stride defect (§3.3) was found and fixed during construction**, not
   inherited. It is called out because the first corpus would have produced a
   complete, plausible benchmark that never exercised the blue perspective
   transform.
2. **Extended batch probing was deliberately conservative.** Only one
   configuration exceeded 2,048 (C2 float16 at 3,072). Per instruction, the host
   was not driven into OOM or swap, so no candidate's maximum batch was
   established by failure. Every "max stable inference batch" of 2,048 in this
   report means "≥ 2,048, not probed to failure", not "2,048 is the ceiling" —
   and with peak memory at 23% for the largest candidate, the true ceilings are
   certainly much higher. If Agent 4 needs a real ceiling, it must be measured
   deliberately.
3. **Boundary C's cost is an upper bound, not a measured positive value.** C − B
   is within noise at every size and negative in two rows. The defensible claim
   is that normalized selection and the inverse frame conversion cost at most a
   few milliseconds at batch 2,048, not that they cost a specific amount.
4. **Boundary B includes the accepted contract validation**, which scans the whole
   observation tensor for finiteness. That is the real
   `tokenize_numpy_observation` path Agent 1 handed over, and it is identical for
   every candidate, but it means boundary B is partly a measurement of the
   current preprocessing implementation rather than of an irreducible cost. It is
   the largest component of the ~10–20 ms B − A gap and is the obvious target if
   Agent 4 finds preprocessing significant for a small candidate.
5. **The 5,000 positions/s floor decided the entire shortlist** (§3.6), since the
   domination clause never fired. It is derived from Phase 3's serial-composition
   arithmetic, but it is a single threshold carrying a lot of weight, and C3
   misses it by 3.8%. This is flagged rather than smoothed over.
6. **`stratego/model/__init__.py` was not modified.** `benchmark_helpers` is
   measurement machinery, not part of the model boundary, so it is imported
   explicitly rather than added to the package's public surface. This is a
   departure from Agent 2's pattern and was chosen to keep the count of modified
   files at zero.
7. **Playing strength was neither measured nor used.** The networks are randomly
   initialised; no game was played, and no win rate exists anywhere in this
   agent's artifacts.
8. **The training step is deliberately incomplete.** No optimizer, no parameter
   update, no scheduler, no hyperparameter selection. Nothing here is training,
   and none of these throughput numbers should be read as a training-speed
   promise for a real loop, which would add optimizer state, data loading and
   checkpointing.

### 3.11 Data files

```text
reports/phase_6_data/agent_03_inference_benchmark.csv        297 rows
reports/phase_6_data/agent_03_training_step_benchmark.csv     56 rows
reports/phase_6_data/agent_03_architecture_shortlist.json
```

Every headline number in this section also exists in those files.

### 3.12 Handoff notes for Agent 4

**ADVANCE — carry these into the integrated Phase 3 pipeline benchmark**

| ID | Configuration | Parameters | Best stable precision | Usable inference batch | Training ex/s @256 | Peak memory |
|---|---|---:|---|---|---:|---:|
| **C0** | 64w × 2 blocks × 4 heads, ff 256 | 123,223 | float16 (+7%) | 256 – ≥2048 | 8,977 | 8.3% |
| **C1** | 128w × 4 blocks × 4 heads, ff 512 | 863,959 | float16 (+21%) | 256 – ≥2048 | 2,966 | 8.2% |
| **C2** | 192w × 4 blocks × 6 heads, ff 768 | 1,922,519 | float16 (+27%) | 256 – ≥3072 | 1,803 | 11.2% |

Recommended starting batch for integrated testing is **1,024**, where all three
sit within 4% of their measured peak (96.3–100%) in both precisions. Batch 512 is
usable but gives up more than the round number suggests — 93.8–98.4% of peak,
with C1 float16 the worst case — so 1,024 is the better default and 512 the
floor if the collector needs the memory. Configuration digests are in the shortlist JSON; rebuild
with `build_candidate_model(candidate_id, seed=20250601)` and require the digest
before timing, exactly as this agent did.

**Numerical caveats.** float16 is numerically sound for all three (max absolute
error ≤ 3.22e-03 on logits, ≤ 1.34e-04 on value probabilities, crafted-margin
agreement exact) but flips 0–2 natural near-ties per 256 positions. If Agent 4
needs bit-reproducible decisions across a checkpoint comparison, use float32; if
it needs throughput, float16 is safe and 7–27% faster.

**The one candidate to consider reinstating: C3.** It is IMPRACTICAL under the
declared rule by a 3.8% float32 margin while its float16 throughput (6,071
positions/s) clears the floor by 21%, with clean numerics and finite gradients.
That measured float16 result is a specific integration reason, and C3 at 2.81 M
parameters is the largest candidate with any claim to practicality. C4, C5 and C6
are 1.5×–4× below the floor and should not be reinstated without a much stronger
measured argument.

**Memory limits.** None were reached. Peak Metal driver allocation was 23.1% of
the 40.2 GB recommended maximum, at C6 / batch 2048 — the largest candidate at
the largest required batch. Agent 4 should expect its constraint to be the
collector's throughput, not the GPU's capacity, and should budget from the
compute figures above rather than from memory.

**What not to reuse.** The corpus here is a *benchmark* corpus: real positions
from uniformly random legal play, with seeded synthetic policy and value targets
that carry no game information. It is correct for timing and for numerical
comparison and wrong for anything that depends on target quality.

## 4. Agent 4 — Integrated Self-Play Pipeline Benchmark

**Status: PASS** — 13 / 13 completion gates true, 0 problems.

Real `stratego_transformer_v1` candidates were inserted into the accepted Phase 3
bulk-synchronous pipeline and measured there. This section does **not** choose
the primary model, and no playing-strength quantity is used as evidence anywhere
in it.

### 4.1 Prerequisite verification

Read from the repository rather than assumed:

| Agent | Status | Commit | Suite at that point |
|---|---|---|---|
| 1 — `model_contract_v2` | `PASS` | `8f4f5e3` | 2,301 passed, 0 failed |
| 2 — candidate family | `PASS` | `8f4f5e3` | 2,383 passed, 0 failed |
| 3 — MPS benchmark | `PASS` | `8f4f5e3` | 2,445 passed, 0 failed |

The architecture family digest is `5b57dd3a0c1a…`, identical to the one Agent 3
recorded, and every candidate was rebuilt from `(candidate id, 20250601)` and
checked against Agent 3's recorded parameter count and serialized configuration
before it was timed. **Architecture modifications: NONE.**

Full suite before any Agent 4 edit, at commit `8f4f5e3`:

```text
python -m pytest -q
2445 passed, 2 skipped, 0 failed in 100.34s
```

That matches Agent 3's recorded end state exactly, so the starting tree was green
and unmodified. The two skips are the pre-existing Phase 4 capability skips.

### 4.2 The integrated candidate set

Agent 3's official ADVANCE list is C0, C1, C2. C3 was **reinstated at review
level for Agent 4**, and the two provenances are recorded separately in
`agent_04_finalists.json` so Agents 5 and 6 can tell them apart.

The reinstatement reason is measured, not preferential: C3 misses Agent 3's
5,000 positions/s practical floor by 3.8% **in float32** (4,811.58), while its
float16 path reaches 6,071 positions/s with clean float16 numerics and a clean
float16 backward pass. The floor was derived from a float32 assumption; the
integrated pipeline runs float16. Agent 3's own handoff independently named C3
as "the one candidate to consider reinstating". C4–C6 remain excluded and were
not measured here.

### 4.3 What changed in the pipeline

`model_contract_v2` puts the model's 10,000 policy logits in the acting player's
normalized squares while the engine stays absolute, so the coordinator has to
convert in both directions. The conversion was added to the coordinator, on the
device, inside the same `_run_chunk` a production step calls.

```text
engine publishes absolute legal product
  -> coordinator permutes it into normalized model legality (device)
  -> candidate forward pass in the normalized frame
  -> masked categorical selection over normalized identifiers
  -> inverse perspective conversion to an absolute engine action
  -> coordinator re-checks the action against the published absolute mask
  -> worker applies the action through the frozen engine
```

Three properties were deliberately preserved:

1. **Phase 3 is unchanged by default.** `CoordinatorConfig.action_frame` defaults
   to `absolute_engine_squares`, so every accepted Phase 3 measurement and test
   means exactly what it meant before. `frame_seconds` is exactly zero there, and
   a test asserts it.
2. **One conversion, one place.** `NormalizedActionFrame` builds both tables by
   calling `stratego.model.action_frame` for all 10,000 identifiers per player.
   It holds no coordinate geometry of its own; a test asserts its tables equal
   Agent 1's per-action functions entry for entry.
3. **One MPS owner.** The model and the frame tables live only in the
   coordinator. `stratego.training.worker_pool` still does not import PyTorch.

Batch conversion is a device `index_select` per acting colour, because Agent 1's
helpers convert one mask at a time and a global step converts up to 2,048 dense
10,000-entry masks. Red's table is the identity under the frozen convention and
is skipped rather than gathered through.

**The trajectory schema did not move.** `trajectory_v1`, format version 1, the
same `DecisionRecord` and `GameRecord` fields, snapshot interval 32. A record
still stores one probability per legal action in ascending **absolute** order;
the coordinator gathers the model's normalized distribution back into that order.
No belief field was added, and none appears in the encoded bytes.

### 4.4 Correctness before timing

An independent set of engine games was advanced in lockstep with the live
pipeline, reusing Phase 3's accepted `ReferenceMirror` comparison unchanged, with
the normalized products re-derived from the *reference* game rather than from the
pipeline's own mask.

| Candidate | Environment steps | Frame rows | Round trips | Red / blue | Illegal | Frame | Model | State/replay |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| C0 | 6,016 | 752 | 752 | 368 / 384 | 0 | 0 | 0 | 0 |
| C1 | 6,016 | 752 | 752 | 360 / 392 | 0 | 0 | 0 | 0 |
| C2 | 6,016 | 752 | 752 | 370 / 382 | 0 | 0 | 0 | 0 |
| C3 | 6,016 | 752 | 752 | 357 / 395 | 0 | 0 | 0 | 0 |

```text
illegal selections       0
action-frame mismatches  0
model/policy errors      0
state/replay mismatches  0
```

Both acting colours were exercised for every candidate. Each sampled row checked
that the normalized legal set has the same cardinality as the engine's and
inverts to it exactly, that the dense normalized mask agrees with the normalized
list, that the published absolute mask is the engine's, and that the selected
normalized identifier is in the normalized legal set and inverts to the absolute
action actually applied.

Non-finite outputs were probed separately on 512 real published positions per
candidate at float16 (5,735,936 logits each, all three heads): **0**. The policy
head is checked explicitly because the contract deliberately permits a model to
score an illegal index arbitrarily and so does not finiteness-check it.

### 4.5 Benchmark topology

The accepted Phase 3 starting point, moved one axis at a time — 56 rows total.

```text
backend            KEEP_PYTHON          precision       float16
CPU workers        10                   live legality   dense
environments       1,536                trajectory      trajectory_v1
MPS owner          coordinator only     snapshot        32
model frame        perspective_normalized_squares
```

Grid rows run with device synchronisation between stages, so the stage fractions
are real; headline rows re-run the best point with those syncs off, so the
throughput figure is not paying for its own instrumentation. Both are in the CSV
with a `timing_mode` column.

### 4.6 Collection-only throughput

Headline rows, 30 s each, timing syncs off:

| Candidate | Parameters | Batch | positions/s | Standalone (Agent 3, float16) | Integrated / standalone |
|---|---:|---:|---:|---:|---:|
| C0 | 123,223 | 2,048 | **17,451** | 27,156 | 64% |
| C1 | 863,959 | 2,048 | **11,875** | 14,919 | 80% |
| C2 | 1,922,519 | 2,048 | **7,495** | 8,636 | 87% |
| C3 | 2,812,247 | 2,048 | **5,496** | 6,071 | 91% |

Every candidate clears the Phase 3 representative probe's integrated figure
(12,838 positions/s) or approaches it, and the *larger* candidates retain a
higher fraction of their standalone rate — the small candidate is the one whose
integrated result is furthest below its hardware ceiling, because at C0's speed
the non-model stages are a larger share of the step.

Where the time goes (batch sweep, synchronised, 1,536 environments):

| Candidate | Batch | positions/s | MPS forward | Host→device | Frame conv. | Norm. legality + sampling | Worker active |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 512 | 15,506 | 0.574 | 0.044 | 0.210 | 0.232 | 0.137 |
| C0 | 1024 | 16,197 | 0.592 | 0.041 | 0.202 | 0.216 | 0.138 |
| C0 | 1536 | 16,310 | 0.597 | 0.039 | 0.213 | 0.222 | 0.132 |
| C0 | **2048** | **18,064** | 0.677 | 0.043 | 0.121 | 0.129 | 0.137 |
| C1 | 512 | 11,615 | 0.781 | 0.031 | 0.076 | 0.091 | 0.088 |
| C1 | 1024 | 11,838 | 0.785 | 0.030 | 0.078 | 0.088 | 0.087 |
| C1 | 1536 | 11,477 | 0.753 | 0.027 | 0.122 | 0.127 | 0.085 |
| C1 | **2048** | **12,161** | 0.797 | 0.028 | 0.071 | 0.077 | 0.089 |
| C2 | 512 | 7,414 | 0.857 | 0.021 | 0.049 | 0.059 | 0.055 |
| C2 | 1024 | 7,550 | 0.867 | 0.019 | 0.047 | 0.054 | 0.056 |
| C2 | 1536 | 7,382 | 0.839 | 0.017 | 0.079 | 0.083 | 0.055 |
| C2 | **2048** | **7,673** | 0.873 | 0.018 | 0.044 | 0.048 | 0.056 |
| C3 | 512 | 5,326 | 0.888 | 0.016 | 0.036 | 0.044 | 0.044 |
| C3 | 1024 | 5,452 | 0.902 | 0.014 | 0.034 | 0.039 | 0.041 |
| C3 | 1536 | 5,394 | 0.887 | 0.049 | 0.049 | 0.052 | 0.042 |
| C3 | **2048** | **5,501** | 0.906 | 0.013 | 0.031 | 0.034 | 0.041 |

**Batch 2,048 is best for every candidate**, and one chunk per step (1,536
environments in a single dispatch) is why: at 2,048 the whole ready set goes to
the device at once, and the frame conversion drops sharply because it is applied
to one contiguous block rather than to two. Batch 1,536 is *worse* than 1,024 for
C1, C2 and C3 despite being larger, which is the same effect seen from the other
side. This is a change from Agent 3's recommendation of 1,024 as the integrated
starting batch; that recommendation was made from standalone curves that could
not see the chunking interaction.

**The frame conversion is the one real cost `model_contract_v2` adds**: 12.1% of
wall time for C0 at its best point, falling to 3.1% for C3, because it is a fixed
per-position cost against a growing model cost. It is not free and it is not
alarming. It is not optimised here (an optimised backend is out of scope), but it
is the obvious first target should Agent 6 want C0-class throughput.

The environment axis is nearly flat for the three larger candidates: 512 → 2,048
environments moves C1 by 5.9%, C2 by 4.1% and C3 by 3.2%. C0 is the exception at
17.7% (14,692 → 17,288), which is consistent with everything else here — it is
the only candidate fast enough for the per-step fixed costs to matter, and more
environments amortise them over more positions. Worker sensitivity is small
— 6 → 14 workers moves throughput by under 6% for every candidate, while
worker-active fraction halves. **The CPU/coordinator balance has inverted since
Phase 3**: workers are now idle 86–96% of the time, where Phase 3's probe left
them 73% busy. No broad CPU scaling study was run, per instruction.

### 4.7 Production recording

The real compact path: `trajectory_v1`, snapshot interval 32, sparse
policy/value decision fields, no 127-channel observations and no dense 10,000
policy vectors stored.

| Candidate | Batch | positions/s | vs collection-only | Recording cost |
|---|---:|---:|---:|---:|
| C0 | 2,048 | **12,689** | 17,451 | −27.3% |
| C1 | 2,048 | **9,420** | 11,875 | −20.7% |
| C2 | 2,048 | **6,486** | 7,495 | −13.5% |
| C3 | 2,048 | **4,886** | 5,496 | −11.1% |

Recording costs less proportionally as the model grows, for the same reason the
frame conversion does. The trajectory write itself is a small share of worker
time (1.4–3.6%); most of the difference is the coordinator's host-side compact
legality pass, which is skipped entirely when recording is off.

### 4.8 Reconstruction

Two independent layers, both required to be exact.

**Live verification (in-worker).** A sampled game carries digests from ply 0 to
its terminal state; the sealed record is round-tripped through the codec and
every decision rebuilt and compared on state fingerprint, observation, absolute
legal list, dense legal mask, belief target, public knowledge, acting player,
selected action and identity triple.

**Stored-game reconstruction (Agent 4's v2 layer).** Encoded records are decoded,
re-encoded and required to be byte-identical, structurally validated, and then
every sampled decision is rebuilt from the nearest snapshot plus replayed actions
and checked on all seven products the instruction names — including the two the
record deliberately does *not* store: the normalized model legal actions and the
selected normalized action, both derived through Agent 1's converter and required
to invert back.

| Candidate | Live decisions / games | Live mismatches | Stored decisions / games | Stored mismatches | Policy re-evaluation deviation |
|---|---:|---:|---:|---:|---:|
| C0 | 746 / 12 | **0** | 3,081 / 40 | **0** | 9.50e-05 |
| C1 | 862 / 14 | **0** | 2,472 / 40 | **0** | 0.00 |
| C2 | 315 / 9 | **0** | 1,954 / 40 | **0** | 0.00 |
| C3 | 182 / 8 | **0** | 1,424 / 40 | **0** | 0.00 |

The last column is a stronger check than exact recovery. Each stored decision was
re-run through the same candidate on the rebuilt observation, masked with the
normalized legal set, and gathered back into ascending absolute order. A frame
error would move probability mass onto entirely different moves and show a
deviation near 1. The measured maximum is 9.5e-05, which is float16 batch-shape
noise. **The recorded distribution is provably the candidate's own distribution,
read through the frame.**

No belief target appears in any encoded record; the bytes were searched for one.

### 4.9 Bottleneck ratio and the backend decision

\[
R=\frac{\text{sustainable simulation capacity}}{\text{sustainable candidate inference capacity}}
\]

**Numerator**, measured here and candidate-independent: the same worker pool at
10 workers × 1,536 environments with the model removed and Phase 3's
deterministic benchmark policy in its place, covering observation building,
legality generation, the engine transition, shared-memory transport, worker
synchronisation and independent reset — **91,778 positions/s** over 25 s (75,761
positions/s with recording enabled).

**Denominator**, measured per candidate: the coordinator's own `_run_chunk` —
host-to-device transfer, the absolute→normalized legality permutation, the
forward pass, masked sampling, the normalized→absolute inverse and the readback —
driven over a real captured batch with the worker pool shut down, for 10 s. It is
literally the same code path a global step executes, not a separately written
imitation.

| Candidate | Numerator | Denominator | R | Serial ceiling | Measured collection | Decision |
|---|---:|---:|---:|---:|---:|---|
| C0 | 91,778 | 22,345 | **4.11** | 17,970 | 17,451 | `KEEP_PYTHON` |
| C1 | 91,778 | 13,264 | **6.92** | 11,589 | 11,875 | `KEEP_PYTHON` |
| C2 | 91,778 | 8,036 | **11.42** | 7,389 | 7,495 | `KEEP_PYTHON` |
| C3 | 91,778 | 5,683 | **16.15** | 5,351 | 5,496 | `KEEP_PYTHON` |

**`KEEP_PYTHON` remains supported, and by a wider margin than Phase 3 measured.**
Phase 3's representative probe gave R = 6.50; a real candidate gives 4.11 at the
smallest and 16.15 at the largest, all far above the 2.0 threshold. No simulator
bottleneck has newly appeared. Only C0 is within a factor of five of the
simulator, and even there the simulator has 4.1× the headroom the model needs.

A useful cross-check: the serial-composition ceiling
\(1/(1/\text{sim}+1/\text{model})\) predicts the measured integrated throughput
to within 3% for every candidate. The pipeline composes as the model says it
should, which is evidence that the stage accounting above is sound.

### 4.10 Storage

**A correction worth stating plainly.** Trajectory bytes are written only when a
game is *sealed*, and a pool starts with all 1,536 environments at ply 0. A short
recording row therefore counts the decisions of ~1,500 unfinished games while
holding almost none of their bytes, and its byte rate is a cold-start transient.
Measured convergence for C0: 0.63 GiB/hour over the first 20 s, rising to a flat
7.0–7.2 GiB/hour from ~1,000 global steps onward. **Dividing whole-run bytes by
whole-run seconds understates the sustained rate by roughly an order of
magnitude**, and the 168-hour projection built that way would have been wrong by
the same factor.

Every figure below therefore comes from a dedicated sustained run: 1,100 global
steps of warmup, then 900 measured steps, sampled every 150 steps. The warmup is
counted in **steps, not seconds**, because desynchronising the slots takes about
two mean game lengths of simulated time — a seconds-based warmup would give the
slowest candidate the least settled measurement. All per-window samples are in
the artifact so the flatness can be checked rather than taken on trust.

| Candidate | GiB/hour | GiB/24 h | GiB/168 h | bytes/decision | Mean game length | Games/s | Window spread | Naive figure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | **7.30** | 175.3 | 1,227 | 186 | 514 | 22.8 | 0.168 | 5.70 |
| C1 | **5.65** | 135.6 | 949 | 188 | 517 | 17.3 | 0.148 | 4.39 |
| C2 | **3.89** | 93.2 | 653 | 186 | 498 | 12.5 | 0.124 | 3.13 |
| C3 | **2.97** | 71.3 | 499 | 187 | 502 | 9.4 | 0.107 | 2.41 |

Two independent consistency checks confirm these are steady-state values.
Bytes/decision lands at 186–188 for **all four** candidates, as it must — it is a
property of the games and the schema, not of the network — and it matches Phase
3's recorded 188. Mean game length lands at 498–517 for all four, likewise. The
transient figures showed neither property, ranging 8–20 bytes/decision and 50–137
plies, which is what exposed the bias.

**Against the user's capacity**, raw 168-hour production:

| Candidate | 168 h | ~150 GB internal | ~1 TB external |
|---|---:|---:|---:|
| C0 | 1,227 GiB | 878% — does not fit | 132% — does not fit |
| C1 | 949 GiB | 680% — does not fit | 102% — does not fit |
| C2 | 653 GiB | 467% — does not fit | 70% — fits |
| C3 | 499 GiB | 358% — does not fit | 54% — fits |

**No candidate's uncompressed week fits internal storage**, and C0 and C1
overflow the external volume as well. A separate probe measured what the codec's
existing compressed path would buy, on 60 real sealed games of production length
(25,015 decisions, mean 417 decisions/game): **ratio 0.685, a 31.5% saving,
186 → 127 bytes/decision**. Compressed, all four fit externally (C0 ≈ 840 GiB).
The pipeline still writes uncompressed records; this is measured so Agent 6 can
decide with a number rather than an assumption.

**The retention policy is deliberately not finalized here.** The user's
preference to keep most games on the external volume is preserved and carried to
Agent 6, and it is now a firm constraint rather than a preference: at C0 or C1,
internal-only retention of a full week is not possible at any snapshot interval.

### 4.11 Finalists

Recommended under a rule stated before the numbers existed and implemented in
`recommend_finalists`. Exclude the numerically unstable and anything that could
not sustain recording; drop candidates dominated on capacity, recording
throughput and memory together; then take the smallest, the largest and the
best-throughput middle of the surviving frontier, so the handoff **spans** the
capacity range rather than clustering at one end.

**Playing strength is not an input.** `FINALIST_INPUT_KEYS` is the complete list
of fields the rule can read, a test asserts no key matches a strength-shaped
substring, and a second test adds a `win_rate` field to every summary and asserts
the chosen finalists do not change.

| Candidate | Parameters | Collection | Recording | GiB/h | R | Process RSS | Metal | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **C0** | 123,223 | 17,451 | 12,689 | 7.30 | 4.11 | 4.19 GiB | 3.61 GiB | **FINALIST** |
| **C1** | 863,959 | 11,875 | 9,420 | 5.65 | 6.92 | 4.28 GiB | 3.15 GiB | **FINALIST** |
| C2 | 1,922,519 | 7,495 | 6,486 | 3.89 | 11.42 | 4.31 GiB | 3.15 GiB | frontier, not selected |
| **C3** | 2,812,247 | 5,496 | 4,886 | 2.97 | 16.15 | 4.31 GiB | 3.15 GiB | **FINALIST** |

**Finalists: C0, C1, C3.**

- **C0** — the throughput end. 17,451 positions/s collecting, 12,689 recording,
  and the only candidate whose integrated rate is far below its hardware ceiling
  (64%), so it is also the one that would gain most from optimisation. Its cost
  is storage: 1,227 GiB a week, the only candidate that overflows the external
  volume even before C1 does.
- **C1** — the balanced middle. 7× C0's capacity for 68% of its collection
  throughput, 5.65 GiB/hour, and the highest integrated efficiency per parameter
  in the set.
- **C3** — the capacity end, and the reinstatement vindicated. 2.81 M parameters,
  22.8× C0's capacity, still sustaining 4,886 positions/s while recording, at the
  lowest storage cost of the four (499 GiB a week, comfortably external). It
  retains 91% of its standalone rate — the highest of any candidate — and R =
  16.15 leaves the simulator with enormous headroom.

C2 is on the frontier and is not eliminated; it is simply not one of the three
spanning picks, sitting between C1 and C3 on every axis. Its full row is in the
artifacts and Agent 6 may reach for it.

All four candidates are numerically stable in the real pipeline, so no candidate
was rejected for instability. C4, C5 and C6 were not advanced and were not
measured.

### 4.12 Deviations and limitations

1. **Two defects in this agent's own instrumentation were found and fixed before
   the reported numbers were produced**, and both are worth naming because they
   would each have shipped a wrong headline.
   - The normalized-selection readback initially sat between two timers, charged
     to neither, while being the first synchronising call — so it silently
     absorbed the sampling kernels' device time and understated
     `normalized_legality_sampling_fraction`. Every device-to-host copy is now
     inside one timed region.
   - The storage rate was initially computed by dividing a 30 s row's bytes by
     its duration, which understated the sustained rate ~11× (§4.10).
2. **Retained-record sampling was biased and is now a reservoir.** `retain_games`
   originally kept the *first* games each worker sealed, which are the shortest
   games of the run — a 10-decision sample where production games run to ~500.
   That made the compression probe meaningless and left the stored-game
   reconstruction layer barely exercising multi-snapshot replay. Note this never
   affected the primary reconstruction evidence: the live verification layer
   selects games at their start and follows them to completion, so it always
   covered full-length games. After the fix the stored sample rose from ~250 to
   1,424–3,081 decisions per candidate, still at zero mismatches.
3. **Game-completion columns in the short grid rows are transient**, and are
   flagged `game_metrics_are_steady_state = false` in the CSV. Only
   `positions_per_second` is sound in those rows; it does not depend on sealing.
   Steady-state game length, games/s and storage come from §4.10's sustained runs.
4. **Batch 2,048 beats Agent 3's recommended 1,024** for all four candidates, a
   chunking interaction standalone curves could not show. Reported rather than
   silently adopted.
5. **The frame conversion is unoptimised.** 12.1% of wall time at C0's best
   point. Building an optimised backend is out of scope by instruction.
6. **Compression is measured but not enabled.** The pipeline writes uncompressed
   records; §4.10's ratio is evidence for Agent 6, not a change to the path.
7. **Random weights throughout.** These are cost measurements. Game lengths,
   terminal-reason mixes and storage rates come from untrained networks and will
   change as a real model learns; no playing-strength claim is made or used.
8. **Not a soak.** The longest continuous run here is ~5.5 minutes. Agent 6 owns
   the one-hour stability soak.

### 4.13 Data files

```text
reports/phase_6_data/agent_04_integrated_pipeline.csv     56 rows
reports/phase_6_data/agent_04_storage_rates.csv            4 rows, steady state
reports/phase_6_data/agent_04_bottleneck_ratios.csv        4 rows
reports/phase_6_data/agent_04_correctness_gate.json
reports/phase_6_data/agent_04_reconstruction.json
reports/phase_6_data/agent_04_compression_probe.json
reports/phase_6_data/agent_04_finalists.json
```

Every headline number in this section also exists in those files.

### 4.14 Completion gates

| Gate | Result |
|---|---|
| Agents 1–3 PASS verified | **true** |
| Real advancing models used, unmodified | **true** |
| v2 correctness run: 0 illegal / frame / model errors | **true** |
| Collection-only benchmark completed fairly | **true** |
| Production-recording benchmark completed | **true** |
| Reconstruction sample: 0 mismatches | **true** |
| Storage rate measured | **true** |
| Candidate-specific R values computed | **true** |
| `KEEP_PYTHON` status explicitly reassessed | **true** |
| Headline runs: 0 unexplained worker/model failures | **true** |
| Two or three finalists identified | **true** |
| No random-weight strength used | **true** |
| Full suite green | **true** |

Commands run:

```text
python scripts/run_phase6_agent04.py
python scripts/run_phase6_agent04_compression.py
python -m pytest -q     ->  2491 passed, 2 skipped, 0 failed in 113.77s
```

Total acceptance runtime 3,414 s. Files created:
`stratego/training/phase6_pipeline_benchmark.py`,
`scripts/run_phase6_agent04.py`,
`scripts/run_phase6_agent04_compression.py`,
`tests/training/test_phase6_candidate_pipeline.py` (46 tests), and the seven data
files above. Files modified: `stratego/training/coordinator.py` (normalized frame,
model injection, `frame_seconds`), `stratego/training/worker_pool.py` (snapshot
counter, reservoir retention), `stratego/training/__init__.py` (docstring).

### 4.15 Handoff notes for Agent 5

**Finalist checkpoints and configuration.** C0, C1, C3, each fully determined by
`(candidate_id, seed=20250601)`:

```python
from stratego.model.production_model import build_candidate_model
model = build_candidate_model("C1", seed=20250601, device="mps", dtype=torch.float16)
```

Require the configuration digest before use, exactly as this agent did. Digests
are in `agent_04_finalists.json` under `candidate_reconstruction`.

**Model construction / loading API.** `stratego.model.checkpoint`, architecture id
`stratego_transformer_v1`, contract `model_contract_v2`. A `model_contract_v1`
checkpoint is refused rather than reinterpreted.

**Normalized neural policy adapter.** `stratego.model.policy_adapter`, with
`stratego.model.action_frame` as the only frame converter. For batch work, the
device-side equivalent is
`stratego.training.coordinator.NormalizedActionFrame`, which builds its tables
from that same module — do not write a third conversion.

**Recommended inference precision: float16.** Numerically clean for all four
candidates in the real pipeline (0 non-finite outputs across 5.7 M logits per
candidate) and 7–27% faster than float32 per Agent 3. Use float32 if
bit-reproducible decisions across a checkpoint comparison are needed; Agent 3
measured 0–2 natural near-tie flips per 256 positions at float16.

**MPS ownership constraints.** The coordinator is the only process that imports
PyTorch or touches Metal; simulation workers are pure NumPy/engine processes and
a test enforces it. Do not move inference into a worker.

**Recommended topology for Agent 5.** 10 workers, 1,536 environments, inference
batch **2,048** (not 1,024), dense legality, float16. Expect the workers to be
idle most of the step — the balance has inverted since Phase 3 and the model, not
the simulator, sets the pace.

**For Agent 6's performance/storage table**, the complete finalist figures are
§4.9, §4.10 and §4.11, and in machine-readable form throughout
`agent_04_finalists.json`. The binding constraint on the 168-hour run is storage,
not compute: at C0 a raw week is 1,227 GiB against ~1 TB external.

---

## 5. Agent 5 — Checkpoint-Aware Parallel Neural Evaluation

**Status: PASS** — 15 / 15 completion gates true, 0 problems.

A Phase 6 finalist checkpoint (C1) now plays Phase 4 evaluation schedules across
CPU game-worker processes while one long-lived Metal inference owner holds the
only model instance. The same 96-match schedule at 1, 2, 4 and 8 workers, and
again with its input order shuffled, produced **one results digest, one
replay-digest set and zero field-level differences**, with the checkpoint loaded
**once**.

The throughput result is stated plainly in §5.10 and is not the one the objective
hoped for: **worker-count scaling does not buy throughput at C1.** The
measurement says why, and the answer is useful for Agent 6.

### 5.1 Prerequisite verification

Read from the repository rather than assumed:

| Agent | Status | Commit | Suite at that point |
|---|---|---|---|
| 1 — `model_contract_v2` | `PASS` | `8f4f5e3` | 2,301 passed, 0 failed |
| 2 — candidate family | `PASS` | `8f4f5e3` | 2,383 passed, 0 failed |
| 3 — MPS benchmark | `PASS` | `8f4f5e3` | 2,445 passed, 0 failed |
| 4 — integrated pipeline | `PASS` | `8f4f5e3` | 2,491 passed, 0 failed |

Agent 4's finalists are C0, C1, C3. **C1 was used**, as instructed. Full suite
before any Agent 5 edit:

```text
python -m pytest -q
2491 passed, 2 skipped, 0 failed in 109.41s
```

### 5.2 The checkpoint

C1 was rebuilt from `(candidate_id="C1", seed=20250601)` and written to
`checkpoints/phase6_c1.pt` (3,478,213 bytes). Its identity was verified against
Agent 4's record *before* it was used for anything:

| Field | Value |
|---|---|
| Parameters | **863,959** — matches Agent 4 exactly |
| Configuration digest | `31ca84ab140c…` — matches Agent 4 exactly |
| `model_contract_version` | `model_contract_v2` |
| `checkpoint_format_version` | 1 |
| `policy_action_frame` | `perspective_normalized_squares` |
| `engine_action_frame` | `absolute_engine_squares` |
| State-dict digest | `46b6ae8f2c00…` — reproducible from `(C1, 20250601)` |
| File digest | `a36e3c5c72bd…` — this write only; see below |

The reload goes through `load_checkpoint(..., expected_architecture_id=…,
expected_configuration=candidate_config("C1"))`, so a file that is merely
self-consistent is not enough: two candidates can share every tensor shape, and
only the configuration distinguishes them.

The two digests answer different questions and only one of them is stable.
`state_dict_digest` covers the weights and is reproducible from
`(candidate_id="C1", seed=20250601)` on any machine — that is the one to compare
against. `checkpoint_file_digest` covers the file's bytes, and
`save_checkpoint` stamps a `creation_timestamp` into every payload, so rewriting
the identical model produces a different file digest. Use it to identify *this
file*, never to check that two people built the same network.

The weights are the family's fixed random initialization. **No playing-strength
quantity anywhere in this section is evidence for anything.**

### 5.3 Topology and MPS ownership

```text
8 CPU game workers            spawn, pure engine/NumPy, no torch import
  -> observer-safe request     identity, decision seed, observation, legality
  -> 1 long-lived owner        the only process holding Metal
     -> checkpoint loaded once
     -> forward pass
     -> deterministic selection in the normalized frame
  -> absolute action back
  -> the worker's engine validates and applies it
```

Measured, not asserted:

| Property | Value |
|---|---|
| Checkpoint loads, greedy owner | **1** (across 5 runs and 85,940 decisions) |
| Checkpoint loads, sampled owner | **1** |
| Checkpoint loads inside game workers | **0** |
| Loads per game / per move | **0 / 0** |
| Game workers importing torch | **0** |
| `stratego.model` modules in a game worker | **none** |
| Checkpoint load time | 0.035 s |

`spawn` is used rather than `fork`, so no child can inherit the parent's Metal
context. Every worker reports what it actually imported and the run counts the
offenders, because the property is easy to lose by accident: `spawn` re-imports
the *launcher's* `__main__` in every child, so a harness that imports torch at
module scope silently puts a PyTorch runtime in every "CPU-only" worker. That is
why `scripts/run_phase6_agent05.py` keeps all torch imports inside functions, and
why the answer is measured rather than assumed.

### 5.4 The observer-safe payload

`InferenceRequest` carries exactly nine fields and nothing else:

```text
request_id  match_id  paired_unit_id  ply  acting_player  decision_seed
observation (127,10,10 float32)  legal_actions (absolute)  legal_action_mask
```

Audited on a request captured from a real game:

| Check | Result |
|---|---|
| Types reachable in the whole object graph | `InferenceRequest`, `str`, `int`, `tuple`, `ndarray` |
| Forbidden types reachable | **none** |
| Arrays aliasing engine-owned memory | **no** — both arrays are copied on construction |
| Privileged class named in the pickled bytes | **no** |
| `stratego.engine.state` named in the pickled bytes | **no** |

Never transported: `GameState`, `PieceRecord`, hidden true identities,
privileged belief targets, true opponent setup, privileged replay object. The
worker-side policy declares `observation=True, legal_action_mask=True,
public_view=False` and nothing else, so nothing else is ever materialised.

### 5.5 Why the worker count cannot change a game

Two independent reasons, and both are needed.

**The game inputs** were already worker-count-independent, and Agent 5 did not
touch that: `MatchSpec` fixes the setups, the colour assignment and both policy
seeds before dispatch, and `derive_decision_seed(policy_seed, ply)` fixes each
decision's stream from the ply alone.

**The model input** is the new risk, and it is where a batching design can
quietly break reproducibility. The default batch policy is `single_request`:
every decision gets its own forward pass, built from that decision's request and
nothing else. The logits — and therefore the action — are a pure function of the
request, so worker count, chunking, arrival timing and schedule order change only
*when* a request is served, never *what* is computed for it. Nothing in this
design assumes approximate float batch equivalence.

Request ordering is canonical regardless: whatever the owner has drained is
sorted by `(match_id, ply, acting_player, request_id)` before being served, so
the sequence of forward passes for a given set of pending requests is fixed by
identity rather than by arrival.

Seeded categorical sampling uses `random.Random(decision_seed)` built per
decision from the Phase 4 seed. No global random stream is consumed in arrival
order.

### 5.6 The greedy reproducibility sweep — the headline gate

96 matches: C1-greedy at float32 against `basic_heuristic`,
`tactical_rule_based` and `strategic_rule_based`, 16 setup pairs each, both
colours (48 paired units, 34,421 plies, 17,188 neural decisions, mean 358.6
plies per match).

| Run | Workers | Chunks | Wall clock | Results digest |
|---|---:|---:|---:|---|
| 1 | 1 | 1 | 76.7 s | `73c4b7bc3575…` |
| 2 | 2 | 8 | 85.1 s | `73c4b7bc3575…` |
| 3 | 4 | 16 | 91.6 s | `73c4b7bc3575…` |
| 4 | 8 | 32 | 105.9 s | `73c4b7bc3575…` |
| 5 | 8, **input shuffled** | 32 | 103.9 s | `73c4b7bc3575…` |

Required outcome, all met:

```text
1 distinct results digest          measured: 1
1 distinct replay-digest set       measured: 1
0 field-level mismatches           measured: 0
```

Per-field mismatch counts across all five runs, every one **zero**:

| Field | Mismatches |
|---|---:|
| `match_id` | 0 |
| `paired_unit_id` | 0 |
| `red_setup` / `blue_setup` | 0 / 0 |
| `candidate_seed` / `opponent_seed` | 0 / 0 |
| `action_history` (absolute) | 0 |
| `replay_digest` | 0 |
| `winner` | 0 |
| `terminal_reason` | 0 |
| `plies` | 0 |

Also identical across all five runs: the full `summarize_run` output, including
the paired bootstrap intervals. 0 policy errors, 0 illegal actions. 16 sampled
rows were replayed through the engine from their stored histories: **0 problems**.

The whole sweep was served by **one** owner holding **one** loaded checkpoint.

### 5.7 The remote path is the serial adapter's path

The strongest correctness statement available here is not "the two agree" but
"there is only one implementation". `NeuralCheckpointPolicy.decide` was
refactored into two pure halves — `prepare_legality` and `select_action` — and
the inference owner calls *those*, so the legality cross-check, both frame
conversions, the greedy tie-break, the categorical sampler and the "converted
back to an illegal action" refusal are the same code Phase 5 runs. There is no
second frame converter and no second selection rule.

That is then checked from the outside. The unmodified Phase 5 adapter, in this
process, on Metal, through the frozen Phase 4 `run_schedule`:

```text
serial adapter, 96 matches    digest 73c4b7bc3575…   87.3 s
owner + 2 game workers        digest 73c4b7bc3575…  101.0 s
field differences: 0
```

### 5.8 Seeded categorical reproducibility

32 matches, C1-sampled against two baselines, 9,566 plies, 4,776 decisions, at 1,
4 and 8 workers plus a shuffled input order:

```text
field-level mismatches      0
distinct results digests    1   (5f645cef0730…)
distinct replay-digest sets 1
checkpoint loads            1
```

And the stochastic path is genuinely not the greedy branch. Over the 32 positions
schedulable both ways:

| Comparison | Result |
|---|---:|
| Matches with different action histories | **32 / 32** |
| Matches with different outcomes | **31 / 32** |
| Results digests differ | **yes** |

### 5.9 Failure behaviour

Twelve cases, each checked for two things: that it is loud, and that **no action
came back**.

| Case | Outcome | Substituted a move |
|---|---|---|
| Missing checkpoint | refused at owner construction | no |
| Incompatible checkpoint — `model_contract_v1` file | `CheckpointError` | no |
| Incompatible checkpoint — C1 file requested as C3 | `CheckpointCompatibilityError` | no |
| Corrupted (truncated) checkpoint | refused | no |
| Malformed request — 7 variants | `InferenceFailure` each, no action returned | no |
| Owner survives a malformed request | next good request answered normally | n/a |
| Non-finite model output on a legal action | `InferenceFailure`, "non-finite" | no |
| Normalized selection converting to an illegal absolute action | `InferenceFailure`, "did not declare legal" | no |
| Inference coordinator failure mid-run | run aborted, `NeuralEvaluationError` | no |
| Timeout / disconnect | `PolicyFailure` wrapping `RemoteInferenceError` | no |
| Crossed response (answer for another request) | refused | no |
| Phase 4 quarantine semantics | `error` result, **no score**, `terminal_reason=policy_error` | no |

The seven malformed-request variants are: wrong observation shape, non-finite
observation, unknown acting player, wrong mask length, empty legal-action list,
negative decision seed, and a mask that disagrees with the legal-action list.

**Nothing anywhere substitutes a random legal, first legal or previous action.**
Every failure path raises or quarantines, and Phase 4's fail-fast/quarantine
semantics are preserved unchanged: a quarantined match carries no score and
`stratego.evaluation.statistics` still refuses to summarise a result set
containing one unless the caller acknowledges it.

### 5.10 Throughput, memory, and the scaling result

**The honest headline: adding game workers makes this slower, not faster.**

| Workers | Wall clock | Matches/s | Positions/s | Decisions/s | Queue wait (mean) | Worker CPU utilisation |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 76.7 s | 1.252 | 449.0 | 224.2 | — | — |
| 2 | 85.1 s | 1.128 | 404.6 | 202.0 | 4.5 ms | 5.0% |
| 4 | 91.6 s | 1.049 | 376.0 | 187.7 | 9.9 ms | 2.7% |
| 8 | 105.9 s | 0.907 | 325.2 | 162.4 | 22.3 ms | 1.4% |

Why, measured rather than guessed. The owner reports both the time it spends on
a decision in total and the forward pass alone, so the same 96-match runs split
cleanly:

| Workers | Wall | Inside the owner | Outside the owner | Owner ms/decision | Forward ms | Owner CPU ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 76.7 s | 70.5 s (92%) | 6.20 s | 4.10 | 1.91 | 2.19 |
| 2 | 85.1 s | 83.3 s (98%) | 1.76 s | 4.85 | 2.54 | 2.31 |
| 4 | 91.6 s | 90.4 s (99%) | 1.18 s | 5.26 | 2.86 | 2.40 |
| 8 | 105.9 s | 104.6 s (99%) | 1.22 s | 6.09 | 3.40 | 2.69 |

Three things follow, and they are the useful part of this section:

1. **The owner is 92–99% of the wall clock, even at one worker.** Game
   simulation — the engine plus the baseline opponent — is 6.2 s of a 76.7 s run
   at one worker, and 1.2 s once it is spread over eight. Parallelising it
   therefore has almost nothing left to parallelise.
2. **Every decision must cross one serial owner.** Each worker blocks on its own
   synchronous request, so eight workers do not get eight forwards at once; they
   queue. The added workers buy idle time (1.4% CPU utilisation at eight) and
   cost queue latency.
3. **Roughly half the owner's per-decision cost is CPU, not GPU.** At C1/float32
   on Metal a decision is 1.9 ms of forward pass and 2.2 ms of legality
   cross-checking, frame conversion and selection — all of it single-threaded in
   the owner, and the CPU half barely moves with worker count (2.19 → 2.69 ms).
   A standalone batch-1 C1 forward is 1.54 ms, so the model itself is not where
   the headroom is.

Two performance instruments, neither of which gates anything:

| Instrument | Wall (24 matches, 8 workers) | Positions/s | vs baseline | Decision agreement |
|---|---:|---:|---:|---|
| float32, `single_request` (the gate path) | 21.5 s | 363.0 | — | — |
| float32, `arrival_batched` (max 8) | 12.5 s | 623.3 | **1.72×** | identical, 0 field differences over 24 matches |
| float16, `single_request` | 22.4 s | 362.3 | **0.96×** | separate policy identity |

Batch sizes actually achieved under `arrival_batched`: mean 3.20, max 7
(histogram 1:198, 2:438, 3:129, 4:77, 5:132, 6:225, 7:16).

**Float16 is slower than float32 here.** At batch 1 the forward is
kernel-launch-bound, not arithmetic-bound, so halving the precision buys nothing
and costs conversion. Agent 3's 7–27% float16 advantage was measured at batch
2,048; it does not survive to batch 1.

**Batching is the only real lever, and it works** — but it is not gate-eligible,
because its batch *membership* depends on which workers happened to be waiting.
A direct probe on eight requests captured from a real game says the arithmetic
would allow it on this hardware:

| Batch size | Bitwise-identical logit rows | Max abs logit difference | Selected-action flips |
|---:|---:|---:|---:|
| 2 | 2 / 2 | 0.0 | 0 |
| 4 | 4 / 4 | 0.0 | 0 |
| 8 | 8 / 8 | 0.0 | 0 |

That is reported as **evidence, not as a licence**. Phase 6's common contract is
explicit that approximate float batch equivalence does not guarantee identical
actions in near-tie positions, and the instruction for this agent was to
prioritise deterministic ordering over throughput. `single_request` therefore
remains the only batch policy any gate in this project runs under.

Memory:

| Quantity | Value |
|---|---:|
| Parent process RSS (the only Metal holder) | 442.1 MB |
| Metal driver-allocated, after the greedy sweep | 51.2 MB |
| Metal current-allocated, after the greedy sweep | 3.5 MB |
| Metal recommended maximum | 40.2 GB |
| Checkpoint on disk | 3.48 MB |
| Game-worker cost | 11.8 CPU-seconds total across 8 workers over 105.9 s |

### 5.11 A bug found and fixed

`categorical_action` widened its logits to float64 for the cumulative sum with a
single `.to(torch.float64)`. Metal has no float64 dtype at all, and a combined
move-and-cast is performed on the source device, so the call raises `TypeError`
on an MPS tensor. Phase 5 only ever sampled from a CPU model, so the
seeded-categorical neural policy **could not run on the device it was built for**
and nobody had found out.

Fixed by moving to the CPU before widening. Nothing numerical changes on the
existing CPU path — the values are already exact float32 by that point and a
device copy is a copy, not a rounding — and the Phase 5 float32 results are
unaffected. Regression test:
`tests/model/test_legality.py::test_the_sampler_runs_on_metal_logits`, which also
asserts the device does not change the chosen action.

### 5.12 Deviations and limitations

1. **The parallel path is currently slower than the serial one.** It is not a
   throughput win at C1; §5.10 measures why. What it *does* deliver is the thing
   Phase 5 could not do at all: a neural checkpoint evaluated across processes,
   safely, with one Metal owner, one checkpoint load and exact reproducibility.
   Agent 6 should choose worker counts on that basis, not on a scaling curve.
2. **`arrival_batched` is measured but never gated.** It is 1.72× faster and was
   decision-identical here, and C1 is bitwise batch-invariant on this hardware —
   but batch membership depends on arrival, so it stays a measurement.
3. **Float16 was benchmarked, never gated**, exactly as instructed. It also has a
   distinct policy identity (`…@0.2.0+float16`), because a half-precision forward
   is a different decision rule and its results must not be attributed to the
   float32 policy.
4. **Phase 3's 10-worker / 1,536-environment topology was not forced onto
   evaluation.** Evaluation runs whole games to termination through the frozen
   `play_match`; a bulk-synchronous environment array is the wrong shape for it.
   MatchSpec, seeds, pairing, result semantics and reproducibility are unchanged.
5. **The neural policies are deliberately absent from the Phase 4 catalogue.**
   `ALL_POLICY_IDS` still holds exactly the same 10 identifiers, so every audit
   that enumerates "all policies" means what it meant in Phase 4. The neural side
   is passed to `play_match` explicitly instead.
6. **`random_legal` was excluded from the greedy sweep.** Phase 5 measured 1,337
   mean plies for greedy against it — roughly four times the other baselines —
   which would have tripled the sweep cost without widening what it tests. Three
   rule-based opponents and both colours are covered.
7. **The C3 load/inference smoke check was not run.** It was explicitly optional
   and not a completion requirement.
8. **Queue-wait timing uses wall-clock timestamps** across processes, because
   `perf_counter` has no cross-process meaning. It is a diagnostic, not a gate.

### 5.13 Completion gates

| Gate | Result |
|---|---|
| Agents 1–4 PASS verified | **true** |
| Stable finalist checkpoint loads under v2 | **true** |
| Checkpoint loaded once per long-lived inference owner | **true** |
| MPS ownership topology safe and documented | **true** |
| Phase 4 identities and seeds unchanged | **true** |
| Observer-safe payload only | **true** |
| Greedy 1/2/4/8/shuffled sweep: 0 mismatches | **true** |
| One results digest and one replay-digest set | **true** |
| Seeded categorical mode reproduces | **true** |
| Failures are loud and never substitute a move | **true** |
| Throughput and load overhead measured | **true** |
| Remote path matches the serial adapter | **true** |
| Stored histories replay through the engine | **true** |
| No random-weight strength used as evidence | **true** |
| Full suite green | **true** |

### 5.14 Files, tests and commands

Created:

```text
stratego/evaluation/neural_worker.py
scripts/run_phase6_agent05.py
tests/evaluation/test_parallel_neural_checkpoint.py   (66 tests)
checkpoints/phase6_c1.pt
```

Modified:

```text
stratego/model/policy_adapter.py     prepare_legality / select_action extracted;
                                     the float64-on-Metal fix
stratego/evaluation/__init__.py      neural_worker exports
tests/model/test_legality.py         the Metal sampler regression
reports/phase_6_implementation_report.md
```

Commands:

```text
python -m pytest -q                  ->  2491 passed, 2 skipped, 0 failed   (before)
python scripts/run_phase6_agent05.py ->  PASS, 1,120.4 s
python -m pytest -q                  ->  2558 passed, 2 skipped, 0 failed   (after)
```

The 67 added tests cover: owner lifecycle and single checkpoint load, workers
importing neither torch nor `stratego.model`, the observer-safe object graph and
pickled payload, canonical request ordering, worker-count and schedule-shuffle
independence on both CPU and Metal, per-decision seed preservation across the
round trip, crossed and reseeded answers, every failure case in §5.9, and the
batch-invariance probe surface.

### 5.15 Data files

```text
reports/phase_6_data/agent_05_parallel_neural_evaluation.json
reports/phase_6_data/agent_05_greedy_worker_sweep.csv
reports/phase_6_data/agent_05_throughput.csv
reports/phase_6_data/agent_05_failure_cases.json
```

### 5.16 Handoff notes for Agent 6

**The API is stable; evaluation does not need redesigning.**

```python
from stratego.evaluation.neural_worker import InferenceOwner, neural_policy_ref, run_neural_schedule
from stratego.model.architecture_configs import candidate_config

owner = InferenceOwner(
    "checkpoints/phase6_c1.pt",
    decision_mode="greedy",              # or "seeded_categorical"
    device="mps", dtype="float32",
    expected_configuration=candidate_config("C1"),
)
run = run_neural_schedule(matches, bank, owner, worker_count=1)
```

**Build the schedule** with `build_paired_schedule(neural_policy_ref("C1"),
policy_ref(opponent_id), pair_ids)`. The neural ref is deliberately not in the
Phase 4 catalogue, so `build_matchup_schedule` will not resolve it — pass it
directly. One owner should serve a whole sweep; that is what keeps the checkpoint
load count at one.

**Use `worker_count=1` unless you need process isolation.** This is the
counter-intuitive part and it is measured: at C1 the serial path is the fastest
path (76.7 s vs 105.9 s at eight workers). Parallel game workers exist here for
safety and reproducibility, not speed.

**The measured ceiling for a soak.** At C1 / float32 / Metal / `single_request`:

```text
~449 positions/s, ~224 decisions/s, ~1.25 matches/s   (worker_count=1)
1.9 ms model forward + 2.2 ms CPU legality/frame/selection per decision
standalone batch-1 C1 forward: 1.54 ms
```

Scale that by candidate: a C0 evaluation will be faster, a C3 evaluation slower,
and the CPU half of the per-decision cost is candidate-independent, so the gap
between candidates will be narrower than their standalone inference numbers
suggest.

**If evaluation throughput becomes the constraint**, the lever is batching, and
the two things needed for it are recorded here: the 1.72× that batching already
achieves, and the bitwise batch-invariance probe. Getting deterministic batching
would need several games in flight per worker so that batch membership is a
function of the schedule rather than of arrival — a design change, not a tuning
knob, and out of scope for both Agent 5 and Agent 6.

**Do not move inference into a worker.** The owner holds the only Metal context,
and a test enforces that game workers import neither torch nor `stratego.model`.

---

## 6. Agent 6 — Stability Soak, 168-Hour Projection, and Architecture Decision

**Status: PASS** — 21 / 21 completion gates true, 0 problems. Recommended Phase 6
status: **PASS**, subject to the reviewing chat's acceptance.

C1 ran continuously for one hour through Agent 4's production-recording pipeline:
30,351,360 positions, 58,741 games, 19,760 global steps, **zero** illegal
actions, action-frame mismatches, reconstruction mismatches, worker failures,
model/MPS failures and non-finite production outputs, and **zero swap**. The
primary architecture is **C1**; the fallback is **C0**. One real limitation is
carried forward and is stated plainly in §6.16: host resident memory rose ~191
MiB per hour and did not visibly decelerate within the hour. A follow-up probe
localized it to the trajectory-recording path — with recording off the
coordinator is flat at +0.8 MiB/hour — which tells Phase 7 where to look.

### 6.1 Prerequisite verification

Read from the repository rather than assumed:

| Agent | Status | Commit | Suite at that point |
|---|---|---|---|
| 1 — `model_contract_v2` | `PASS` | `8f4f5e3` | 2,301 passed, 0 failed |
| 2 — candidate family | `PASS` | `8f4f5e3` | 2,383 passed, 0 failed |
| 3 — MPS benchmark | `PASS` | `8f4f5e3` | 2,445 passed, 0 failed |
| 4 — integrated pipeline | `PASS` | `8f4f5e3` | 2,491 passed, 0 failed |
| 5 — parallel evaluation | `PASS` | `8f4f5e3` | 2,558 passed, 0 failed |

Full suite before any Agent 6 edit, at commit `7001f75`:

```text
python -m pytest -q
2558 passed, 2 skipped, 0 failed in 149.78s
```

That matches Agent 5's recorded end state exactly. The two skips are the
pre-existing Phase 4 capability skips.

The architecture family digest is `5b57dd3a0c1a…`, unchanged since Agents 2 and
3. Every candidate in the comparison was rebuilt from `(candidate id, 20250601)`
and checked against Agent 2's recorded parameter count **and** configuration
digest before anything was measured:

| Candidate | Parameters | Matches Agent 2 | Configuration digest | Matches |
|---|---:|---|---|---|
| C0 | 123,223 | yes | `057d6c9242e3…` | yes |
| C1 | 863,959 | yes | `31ca84ab140c…` | yes |
| C2 | 1,922,519 | yes | `3f49fc3a7c34…` | yes |
| C3 | 2,812,247 | yes | `62cffd75b6f9…` | yes |

**Architecture modifications: NONE.**

### 6.2 Which candidate was soaked, and why that one

The instruction requires the leading soak candidate to be chosen from Agents 3–4
measured evidence alone. That choice is made by
`stratego.training.phase6_soak.select_architectures`, whose complete input list is
`SELECTION_INPUT_KEYS` — capacity proxy and measured cost, nothing else — and the
rule was written before its output was read. It runs *before* the soak, so the
soak candidate is a derived result rather than a preference.

Walking the ladder in parameter order, each step is scored by

\[
\text{score}=\frac{\log(1+\Delta\text{parameters})}{-\log(1+\Delta\text{recording throughput})}
\]

— capacity bought per unit of sustained production throughput given up:

| Step | Parameters | Recording throughput | Collection | Training step | float16 inference | Storage rate | Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 → C1 | **+601.1%** (7.01×) | **−23.5%** | −32.0% | −66.5% | −45.1% | −22.6% | **6.54** |
| C1 → C2 | +122.5% (2.23×) | −30.4% | −36.9% | −39.1% | −42.1% | −31.3% | 2.21 |
| C2 → C3 | +46.3% (1.46×) | −24.0% | −26.7% | −30.2% | −29.7% | −23.6% | 1.39 |

The first rung buys seven times the capacity for less than a quarter of the
sustained throughput. The very next rung buys 2.2× capacity for a *larger*
proportional throughput loss — a 66% collapse in efficiency — and the one after
that is worse again. **The knee is C1**, and the declared floor of 0.5 is not
load-bearing: any floor above ≈0.35 selects C1, and a test asserts it across
0.4–1.0.

Memory is not a differentiator anywhere on this ladder (process RSS 4.19–4.31
GiB, Metal 3.15–3.61 GiB), and every candidate is numerically stable at float16
in the real pipeline, so neither axis breaks the tie. Storage argues against C0
and mildly for the larger candidates; §6.13 shows it does not overturn the knee.

C1 is therefore the leading finalist and was soaked first. No random-weight game
result was consulted; a test adds a `win_rate`, an `elo` and a `match_score` to
every row and asserts the selection does not move.

### 6.3 What was built

```text
stratego/training/phase6_soak.py       the soak loop, the growth/drift statistics,
                                       the 168-hour projection, the storage
                                       analysis and the selection rule
scripts/run_phase6_agent06.py          the acceptance harness
scripts/run_phase6_agent06_memory.py   the memory-localization probe of §6.16
tests/training/test_phase6_soak.py     74 tests
```

Nothing in `stratego/engine/`, `stratego_project_docs/`, the Phase 4 evaluation
semantics or the trajectory schema was touched. The soak drives Agent 4's
pipeline; it does not re-implement any part of it.

### 6.4 The one-hour soak

Agent 4's best defensible production point, adopted unchanged:

```text
candidate            C1, real weights from (C1, 20250601)
contract             model_contract_v2, perspective_normalized_squares
workers              10                 environments      1,536
inference batch      2,048              precision         float16
live legality        dense              MPS owner         coordinator only
recording            production trajectory_v1, uncompressed
snapshot interval    32                 backend           KEEP_PYTHON
duration             3,600.7 s continuous, 60 samples at 60 s
```

**Why not `run_neural_schedule`.** Agent 5's ~449 positions/s is *evaluation*
throughput — whole games played to termination through `play_match`, one forward
pass per decision at batch 1. The 168-hour run is a collection run, and the
machine it will use is the bulk-synchronous coordinator. Agent 5's figure appears
nowhere in this section's collection or training arithmetic.

**Why not `worker_count=1`.** That recommendation is Agent 5's, and it is about
deterministic Phase 4 evaluation, where every decision crosses one serial
inference owner and extra game workers only add queue latency. Collection is a
different shape: the workers build observations and advance environments in
parallel between barriers. Agent 4's 10-worker topology is the right one here and
is what was soaked.

**Two independent legality authorities ran on every single decision.** The
coordinator checks each sampled action against the very mask it was drawn from
before the workers see it (`verify_sampled_legality`), and the frozen engine
validates every action in `apply_actions` before applying any of them. That is
**30,351,360** action-legality checks, one per position, over the hour.

**Reconstruction ran for the whole hour, not as an opening budget.** Each worker
carries live digests through one game at a time, continuously; a sealed record is
round-tripped through the codec and every decision rebuilt from the nearest
snapshot plus replayed actions, then compared field by field. The budget is set
so it cannot exhaust inside the hour, and a gate asserts the verified count was
still rising at the last sample.

**Non-finite outputs.** `ModelOutputs.validated` refuses a non-finite value or
belief head inside *every* forward pass, so those two heads were checked
continuously and for free across all 19,760 steps. The contract deliberately does
not finiteness-check the policy head — a model may score an illegal index
arbitrarily — so a probe covers it explicitly every 60 s, on real published
positions, through the live model at pipeline precision.

### 6.5 Hard soak targets

| Required | Measured | |
|---|---:|---|
| illegal actions | **0** | over 30,351,360 checks |
| action-frame mismatches | **0** | |
| trajectory reconstruction mismatches | **0** | over 195,686 decisions in 399 games |
| worker failures | **0** | 600 liveness checks, 10/10 alive throughout |
| MPS/model failures | **0** | |
| non-finite production outputs | **0** | 344,156,160 logits probed across all three heads |
| swap | **0** | start, end and every one of the 60 samples |
| unexplained persistent memory growth | see §6.6 | gate passed; a real trend is reported |

`games_joined_late` was 0: no game was recorded from a partial trajectory. The
finiteness probe cost 2.49 s of 3,600.7 s — **0.069%** of wall clock — and is
counted inside the reported throughput rather than subtracted from it.

Terminal reasons over the hour: flag capture 32,155; battleless move-limit draw
25,488; opponent has no legal move 1,097; both players have no legal move 1.
These come from **random weights** and are a cost measurement, not a statement
about how Stratego games end.

### 6.6 Throughput, drift and memory

Two windows, never averaged together. Correctness covers the whole hour; the
sustained rates come from the steps after the warmup.

```text
warmup            3,000 global steps = 560.3 s
measured window   16,760 global steps = 3,039.8 s, 51 samples
```

The warmup is longer than Agent 4's 1,100 steps for a reason the calibration
pilot measured: a pool starts with all 1,536 slots at ply 0 and grows their
trajectory builders in lockstep, so resident memory climbs toward the envelope of
a fully synchronised population before the slots spread out. That climb was still
converging at step ~1,300. Six mean game lengths puts the measurement window past
it.

| Sustained (measured window) | |
|---|---:|
| recording-inclusive positions/s | **8,468.7** |
| games/s | **16.53** |
| trajectory production | **5.358 GiB/hour** |
| bytes/decision | 188.71 |
| mean game length | 512.4 |
| whole-hour positions/s including warmup | 8,430.7 |

**Drift is negligible and slightly negative**: −16.4 positions/s per hour, which
is **−0.19% per hour**, with R² = 0.0017 — the fit explains essentially none of
the variance, so this is scatter rather than decay. Coefficient of variation
across the 51 windows is 1.14% (min 8,295, max 8,649), and the second half of the
window averaged **0.20% faster** than the first. The pipeline did not degrade.

Memory, over the measured window:

| Quantity | First half | Second half | Change | Slope | Within 2%? |
|---|---:|---:|---:|---:|---|
| Metal current allocated | 2,051,072 B | 2,051,072 B | **0.0000%** | 0 | yes |
| Shared memory | 94,288,896 B | 94,288,896 B | **0.0000%** | 0 | yes |
| Metal driver allocated | 2.031 GiB | 2.031 GiB | +0.0007% | +50 KiB/h | yes |
| Coordinator RSS | 4.011 GiB | 4.051 GiB | +1.00% | +95 MiB/h | yes |
| Worker RSS (10 total) | 4.498 GiB | 4.539 GiB | +0.92% | +96 MiB/h | yes |
| **Total RSS** | **8.508 GiB** | **8.590 GiB** | **+0.96%** | **+191 MiB/h** | yes |

Swap was zero at start, at end, and at every sample.

The two device-side quantities are *exactly* constant, to the byte, for the whole
hour — the Metal allocator and the shared-memory block do not move. The growth is
entirely host RSS, and it is discussed as a limitation rather than dismissed in
§6.16.

### 6.7 A correction to Agent 4's recording headline

Agent 4 reported C1 recording at **9,420 positions/s**. That figure comes from a
30-second row on a cold pool: it pays the recording cost of every decision while
almost no game has sealed yet, so it never pays the sealing and encoding cost a
steady-state run pays continuously. Agent 4's own warmed storage run gives
**8,954 positions/s** for the same candidate, and this soak — 3,040 measured
seconds — gives **8,468.7**.

This is exactly the correction Agent 4 made for trajectory *bytes* in its §4.10,
applied to positions/s. Both figures are carried in the artifacts, the selection
rule and the projection read the sustained one, and a test asserts the knee lands
on C1 under either. Agent 4's collection-only and standalone numbers are
unaffected — nothing there depends on sealing.

### 6.8 The finalist comparison

All four measured candidates. **C2 is retained** even though Agent 4's formal
finalists were C0/C1/C3: it is on the measured frontier, it was measured on every
axis, and the knee argument turns on the C1 → C2 step.

| | C0 | **C1** | C2 | C3 |
|---|---:|---:|---:|---:|
| configuration | 64w × 2b × 4h, ff 256 | **128w × 4b × 4h, ff 512** | 192w × 4b × 6h, ff 768 | 192w × 6b × 6h, ff 768 |
| parameters | 123,223 | **863,959** | 1,922,519 | 2,812,247 |
| checkpoint | 0.48 MiB | **3.31 MiB** | 7.35 MiB | 10.75 MiB |
| standalone float32 | 25,363 | **12,304** | 6,785 | 4,812 |
| standalone float16 | 27,156 | **14,919** | 8,636 | 6,071 |
| training step examples/s | 9,084 | **3,046** | 1,854 | 1,294 |
| training step memory | 2.14 GiB / 1.12 GiB | **2.14 GiB / 1.09 GiB** | 2.14 GiB / 2.12 GiB | 2.14 GiB / 2.12 GiB |
| integrated collection | 17,451 | **11,875** | 7,495 | 5,496 |
| recording, Agent 4 headline (cold) | 12,689 | 9,420 | 6,486 | 4,886 |
| recording, Agent 4 sustained | 11,706 | 8,954 | 6,231 | 4,735 |
| recording, Agent 6 one-hour soak | — | **8,468.7** | — | — |
| games/s (Agent 4 sustained) | 22.77 | 17.34 | 12.52 | 9.44 |
| games/s (soak) | — | **16.53** | — | — |
| MPS utilisation | 0.677 | **0.797** | 0.873 | 0.906 |
| worker wait fraction | 0.863 | **0.911** | 0.944 | 0.959 |
| process / shared / Metal (Agent 4) | 4.19 G / 89.9 M / 3.61 G | **4.28 G / 89.9 M / 3.15 G** | 4.31 G / 89.9 M / 3.15 G | 4.31 G / 89.9 M / 3.15 G |
| GiB/hour (Agent 4 sustained) | 7.30 | 5.651 | 3.89 | 2.97 |
| GiB/hour (soak) | — | **5.358** | — | — |
| bottleneck ratio R | 4.11 | **6.92** | 11.42 | 16.15 |
| soak status | not soaked | **one hour, 12/12 gates** | not soaked | not soaked |

Every row states its own source. Agent 4's sustained figures come from warmed
storage runs of 125–308 seconds; the soak rows are this agent's 3,040-second
measured window. C1 is the only candidate with a soak row because it is the only
candidate that was soaked — see §6.16 item 4. The coordinator's own resident
memory during the soak is reported separately in §6.6, because the soak measures
current RSS across the coordinator and all ten workers while Agent 4's column is
a peak-RSS figure for the parent alone; the two are not the same quantity and are
deliberately not merged into one row.

**Parameter count is a capacity proxy, not a proven strength measurement.** Every
candidate here carries the family's fixed random initialization. Nothing in Phase
6 has measured playing strength, and nothing in this decision claims to.

### 6.9 The capacity/compute knee

The knee is C1, for the reasons tabulated in §6.2. Stated as the instruction
frames it — the largest useful capacity increase before additional size costs
disproportionately:

- **C0 → C1 is clearly worth taking.** 7.01× the capacity for 23.5% of the
  sustained recording throughput. C0 is also the candidate furthest below its own
  hardware ceiling (64% of standalone), which means its throughput advantage is
  partly an artifact of the pipeline's fixed per-step costs rather than of the
  model being cheap.
- **C1 → C2 is not.** 2.23× capacity costs 30.4% of recording throughput, 39.1%
  of training-step throughput and 42.1% of standalone inference — a larger
  proportional loss than the previous step, for less than a third of the capacity
  gain. Efficiency falls by 66%.
- **C2 → C3 is worse still.** 1.46× capacity for another 24.0%.

This is not "pick the fastest" (that would be C0, and it is the fallback, not the
primary) and not "pick the largest" (that would be C3). It is the last rung whose
price the measurements justify.

### 6.10 Primary architecture

```text
candidate_id                 C1
architecture_family          stratego_transformer_v1
architecture_family_version  architecture_family_v1
model_contract_version       model_contract_v2
configuration_digest         31ca84ab140c523e65567787b0289fe0dbdf5ab0344667410a5fda7060cfe07d
initialization_seed          20250601

width                        128
blocks                       4
heads                        4                (head dimension 32)
feed_forward_width           512
position encoding            learned_row_column_v1
normalization                pre_layernorm
policy head                  source_query_destination_key_scaled_with_source_and_destination_biases
value head                   mean_pool_tokens_then_two_layer_mlp
belief head                  per_token_linear
parameters                   863,959
checkpoint                   3,473,613 bytes

recommended MPS precision    float16 for collection, float32 for evaluation
recommended inference batch  2,048 for collection; 1 with single_request for evaluation
recommended topology         10 workers x 1,536 environments, dense legality,
                             snapshot interval 32, coordinator is the only MPS owner;
                             evaluation at worker_count=1
```

The two precisions are not an inconsistency. Agent 4 measured 0 non-finite
outputs across 5.7 M logits per candidate at float16 in the real pipeline, and
float16 is faster at collection batch sizes; Agent 5 measured float16 to be
*slower* than float32 at batch 1, where the forward pass is kernel-launch-bound,
and gated evaluation at float32. A half-precision forward is a different decision
rule, and Agent 5 gave it a distinct policy identity for exactly that reason.

**The measured tradeoff.** C1 gives up 27.6% of C0's sustained recording
throughput and two-thirds of its training-step throughput, and buys 7.01× the
parameters. It keeps the simulator at R = 6.92 — the model, not the engine, sets
the pace, with the simulator holding roughly seven times the headroom the model
needs. It produces 1.94 GiB/hour less trajectory than C0, which is what makes a
full week fit the external volume at all (§6.13). It sustained an hour with every
hard gate clean.

### 6.11 Fallback architecture

```text
candidate_id                 C0
architecture_family          stratego_transformer_v1
architecture_family_version  architecture_family_v1
model_contract_version       model_contract_v2
configuration_digest         057d6c9242e328900f923d4e4c265eaba1bf95e57e1be120a024d2c42c143ddd
initialization_seed          20250601

width                        64
blocks                       2
heads                        4                (head dimension 16)
feed_forward_width           256
position encoding            learned_row_column_v1
normalization                pre_layernorm
policy head                  source_query_destination_key_scaled_with_source_and_destination_biases
value head                   mean_pool_tokens_then_two_layer_mlp
belief head                  per_token_linear
parameters                   123,223
checkpoint                   504,965 bytes

recommended MPS precision    float16 for collection, float32 for evaluation
recommended inference batch  2,048 for collection; 1 with single_request for evaluation
recommended topology         identical to the primary
```

This is a frozen exact configuration, reproducible from `(C0, 20250601)`, not an
informal idea. It qualifies on every required ground:

- **Full correctness.** Agent 4's `model_contract_v2` gate: 6,016 environment
  steps, 752 frame rows, 0 illegal selections, 0 frame mismatches, 0 model errors,
  0 state/replay mismatches, both colours exercised; 3,081 stored decisions
  reconstructed with 0 mismatches.
- **Stable MPS behaviour.** Numerically clean at float16 in the real pipeline,
  stable to inference batch 2,048, 0 non-finite outputs.
- **Materially better throughput.** 38.1% more sustained recording throughput
  (11,706 vs 8,954) and 2.98× the training-step rate (9,084 vs 3,046 examples/s).
  The instruction asks for materially better throughput *and/or* memory headroom;
  C0 qualifies on throughput alone. It does **not** qualify on memory: its
  training step uses 1.12 GiB of Metal against C1's 1.09 GiB, and its integrated
  process RSS is 4.19 GiB against 4.28 GiB — the two are within noise of each
  other, and C0's Metal figure is marginally the larger. Memory does not separate
  any candidate on this ladder, and claiming otherwise for the fallback would
  misread the measurements.
- **The same model contract**, so a checkpoint of either loads through the same
  strict path and plays under the same normalized action frame.

Its cost is capacity — 123,223 parameters is 14% of C1's — and storage: 7.30
GiB/hour, which is the one candidate whose raw week does not fit the external
volume (§6.13).

### 6.12 The 168-hour projection

The user's official final run is exactly \(168\ \mathrm{h}=604{,}800\ \mathrm{s}\).
The measured input is this soak's own sustained rate; everything below the line
is arithmetic on it.

**Measured** (C1, 3,039.8 s steady-state window of the one-hour soak):

```text
recording-inclusive throughput   8,468.7 positions/s
games                            16.53 games/s
trajectory production            1,598,143 bytes/s = 5.358 GiB/hour
bytes/decision                   188.71
checkpoint                       3,473,613 bytes
training step (Agent 3)          3,045.75 examples/s
```

**Extrapolated to 604,800 s:**

| | 24 hours | 168 hours |
|---|---:|---:|
| positions | 731,695,002 | **5,121,865,011** |
| games | 1,428,097 | **9,996,679** |
| trajectory | 128.60 GiB | **900.18 GiB** (966.6 GB) |

Checkpoint storage is negligible at any plausible frequency:

| Frequency | Retained | Total |
|---|---:|---:|
| hourly | 168 | 557 MiB |
| every 4 hours | 42 | 139 MiB |
| every 12 hours | 14 | 46 MiB |
| daily | 7 | 23 MiB |

**Training-step opportunities.** Agent 3 measured C1's backward pass at 3,045.75
examples/s *standalone, with no simulator running*. If a training process had the
device entirely to itself for the whole week it could consume 1.84 × 10⁹
examples — about **0.36 epochs** over the 5.12 × 10⁹ positions the same week
would produce. Collection and training contend for one Metal device, so the real
concurrent figure is lower. The useful reading is structural rather than
numerical: **this pipeline produces data considerably faster than a single M4 Pro
can learn from it**, so the final run's design question is sampling and retention,
not how to collect more.

None of this is a claim about learning. These totals follow from a cost rate
measured on random weights; a trained network's game lengths — and therefore its
games/s and its byte rate — will differ.

### 6.13 Storage analysis

Against the user's declared capacity of ~150 GB internal and ~1 TB external:

| Candidate | GiB/hour | 168 h raw | vs internal | vs external | Compressed (×0.685) | vs external |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 7.30 | 1,227 GiB | 878% | **132% — no** | 841 GiB | 90% — yes |
| **C1** | **5.358** ‡ | **900 GiB** | 644% | **96.7% — yes, barely** | **617 GiB** | **66% — yes** |
| C2 | 3.89 | 653 GiB | 467% | 70% — yes | 447 GiB | 48% — yes |
| C3 | 2.97 | 499 GiB | 358% | 54% — yes | 342 GiB | 37% — yes |

‡ measured by this soak; the others are Agent 4's sustained runs.

**The headline result is better than Agent 4's projection suggested.** On Agent
4's 5.651 GiB/hour, C1's raw week came to 1,019 GB — 102% of the external volume,
overflowing it. This soak's longer measurement gives 5.358 GiB/hour and therefore
966.6 GB, which **fits the 1 TB external volume uncompressed, with about 3%
spare**. That is a thin margin and should not be relied on alone.

**No candidate's uncompressed week fits internal storage.** 150 GB holds roughly
26 hours of C1's raw trajectory.

The compression figure is Agent 4's **measured** ratio — 0.685 on 60 real sealed
games of production length, 25,015 decisions, 186 → 127 bytes/decision — not an
assumption. Compressed, C1's week is 662 GB, 66% of the external volume, leaving
room for a second week. The pipeline still writes uncompressed records;
`RecordingConfig.compress_records` already exists and already has a decode path,
so this is a flag rather than new work, and Phase 6 did not enable it.

**Recommended retention approach**, not finalized here:

```text
internal (~150 GB):
  active checkpoint and the current checkpoint ladder   ~0.6 GB even hourly
  hot logs and metrics
  the live replay buffer / current shard being written

external (~1 TB):
  compressed full trajectories for the whole run        ~662 GB at C1
  evaluation and human games
  diagnostic and unusual games
  headroom                                              ~340 GB
```

**Deleting most games is not necessary and is not recommended.** The user's stated
preference to preserve most games externally is achievable in full: every game of
the 168-hour run fits on the external volume, uncompressed with a 3% margin or
compressed with a 34% margin. Phase 3's rolling retention was designed under a
different measurement and should not be carried over by default. If the run is
extended beyond one week, or if the trained network's games run longer than the
random-weight games these rates were measured on, compression should be enabled
before tiering or deletion is considered.

### 6.14 Parallel evaluation readiness

Checked, not asserted. A checkpoint was written for each of C1 and C0 and
reloaded through the same strict path Agent 5 gated on — `load_checkpoint` with
`expected_architecture_id` **and** `expected_configuration`, which refuses a file
that is merely self-consistent — and an `InferenceOwner` was constructed on Metal
in both decision modes for each.

| | C1 (primary) | C0 (fallback) |
|---|---|---|
| loads under expected configuration | yes | yes |
| checkpoint `model_contract_version` | `model_contract_v2` | `model_contract_v2` |
| checkpoint format version | 1 | 1 |
| `policy_action_frame` | `perspective_normalized_squares` | `perspective_normalized_squares` |
| `engine_action_frame` | `absolute_engine_squares` | `absolute_engine_squares` |
| parameters after reload | 863,959 | 123,223 |
| greedy owner | constructs, 1 checkpoint load | constructs, 1 checkpoint load |
| seeded categorical owner | constructs, 1 checkpoint load | constructs, 1 checkpoint load |

**Future checkpoints of either architecture are ready for deterministic 1/2/4/8-worker
evaluation, in greedy mode and in seeded categorical mode.** Agent 5 reproduced
one results digest and one replay-digest set across 1, 2, 4, 8 and shuffled-input
runs with zero field-level mismatches, and nothing in Agent 6 changed the
evaluation path, the batch policy or the model contract those results depend on.

The accepted evaluation reference is unchanged: **float32** with the
**`single_request`** batch policy, `worker_count=1` for speed unless process
isolation is wanted. `arrival_batched` remains experimental performance evidence
only and was not used, extended or redesigned here.

### 6.15 Backend decision

```text
KEEP_PYTHON remains supported
```

Agent 4 measured R per candidate against a candidate-independent simulator
numerator of 91,778 positions/s: C0 4.11, C1 6.92, C2 11.42, C3 16.15, all far
above the 2.0 threshold. For the selected primary the simulator holds roughly
seven times the headroom the model needs. The soak sustained that same regime for
an hour — workers idle ~91% of each step, the model setting the pace — with no
sign of the balance moving. **No simulator bottleneck has appeared, and no
optimized backend is required.** None was built.

### 6.16 Deviations and limitations

1. **Resident memory grew ~180 MiB/hour and did not visibly decelerate within the
   hour. This is the highest-risk remaining limitation.** The declared gate — the
   second half of the measurement window against the first, tolerance 2% — passed
   at +0.96%, and the absolute rise over 50 measured minutes was 157 MiB on an 8.5
   GiB base. But the trend is monotone with R² = 0.96, and split into thirds it
   reads +178, +233, +176 MiB/hour: roughly constant, not decaying. The device
   side is exactly flat — Metal current allocated and the shared-memory block did
   not move by a single byte — so this is host RSS, split about evenly between the
   coordinator and the ten workers.

   **The growth was localized rather than left as a guess.** A follow-up probe
   (`scripts/run_phase6_agent06_memory.py`) reran the identical topology, candidate
   and seed with production recording switched *off*:

   | Settled slope | Recording on | Recording off | Attributable to recording |
   |---|---:|---:|---:|
   | coordinator RSS | +95.1 MiB/h | **+0.8 MiB/h** | +94.3 MiB/h |
   | worker RSS (10 total) | +95.9 MiB/h | +25.9 MiB/h | +70.0 MiB/h |
   | total RSS | +191.1 MiB/h | +26.7 MiB/h | +164.4 MiB/h |

   With recording off the coordinator is flat — **+0.8 MiB/hour** across a
   ten-minute settled window, against +95.1 MiB/hour with recording on — so **the
   collection path is not what grows**, and 164 of the soak's 191 MiB/hour belong
   to the production trajectory path. That is consistent with the
   mechanism: recording is what allocates and frees a ~96 KiB encoded record 16.5
   times a second, plus the per-decision builder objects and the coordinator-side
   compact-legality and probability blocks that exist only when recording is on.
   Churn of that shape produces allocator arena growth, which is bounded in
   principle — the move limit caps a game, and records are dropped immediately
   after encoding — but can creep for a long time before it plateaus.

   This narrows the question without closing it. One hour still cannot prove the
   arena plateaus, and extrapolated naively 191 MiB/hour over 168 hours is ~31 GiB
   on top of the pipeline's 8.6 GiB, which on this 48 GiB machine would approach
   the point where swap becomes a risk — and swap was a zero-tolerance gate for
   good reason. **Recommended mitigations for the final run, in order of cost:**
   monitor RSS and swap continuously; restart the collection process at a
   checkpoint boundary every 24 hours, which the checkpoint-based design already
   makes cheap and which resets the arena outright; and, if Phase 7 wants the
   question settled, run a multi-hour probe against the recording path
   specifically, now that it is known to be the right place to look. Nothing here
   blocks Phase 6 — the hour was clean and swap was zero — but it should not be
   carried into an unattended 168-hour commitment unexamined.

2. **The warmup is longer than Agent 4's.** 3,000 steps rather than 1,100, chosen
   from the calibration pilot because the memory question needs a settled window
   and the pilot showed resident memory still converging at step ~1,300. This only
   makes the sustained rates *more* settled than Agent 4's; the storage figures
   remain directly comparable and land at 188.71 bytes/decision against Agent 4's
   188.24.

3. **Agent 4's recording headline is superseded for C1**, per §6.7. The 9,420
   positions/s figure is a cold-pool number; the sustained rate is 8,468.7. Both
   are in the artifacts. This does not change any finalist, any classification or
   the knee.

4. **Only C1 was soaked.** The instruction requires the leading finalist to
   complete the hour and provides for soaking the next finalist only if it fails.
   C1 passed 12/12, so C0, C2 and C3 carry Agent 4's shorter measurements and are
   marked "not soaked" in the comparison. The fallback's suitability rests on
   Agent 4's correctness gate and benchmarks, not on a soak of its own.

5. **Random weights throughout.** Game lengths, terminal-reason mixes, games/s and
   therefore the storage rate all come from untrained networks and will change as a
   real model learns. No playing-strength claim is made or used anywhere.

6. **Trajectory bytes are produced, not written to disk.** The recording path
   encodes every sealed record and measures its length, then drops it —
   `retain_games` is 0 and the worker pool performs no file I/O. The GiB/hour
   figure is therefore a *production* rate, which is the right input for a storage
   projection, but the final run will additionally pay real write bandwidth and
   filesystem overhead that this soak did not measure.

7. **Compression is measured but not enabled**, exactly as Agent 4 left it. §6.13
   uses Agent 4's measured 0.685 ratio; the pipeline still writes uncompressed
   records.

8. **The stage-timing fractions in the soak are not attributable.**
   `detailed_timing` is off, because a soak should measure what production
   sustains rather than pay for its own instrumentation. Where the time goes comes
   from Agent 4's synchronised grid, not from here.

9. **A harness ordering defect was found and fixed.** The acceptance script
   originally ran its post-change suite *before* writing the artifacts, so the nine
   tests that validate the published numbers skipped silently — they are written to
   skip when the files are absent. The script now writes the artifacts, runs the
   suite, and rewrites the decision with the result. The reported 2,632-passing
   figure is from a run in which those nine tests actually executed.

### 6.17 Completion gates

| Gate | Result |
|---|---|
| Agents 1–5 all PASS | **true** |
| Finalist configurations and parameter counts reproduced | **true** |
| `model_contract_v2` with normalized model actions | **true** |
| One-hour soak completed continuously | **true** |
| Soak illegal actions = 0 | **true** |
| Soak action-frame mismatches = 0 | **true** |
| Soak reconstruction mismatches = 0 | **true** |
| Soak worker failures = 0 | **true** |
| Soak model/MPS failures = 0 | **true** |
| Soak non-finite production outputs = 0 | **true** |
| Swap = 0 | **true** |
| No unexplained persistent memory growth | **true** (declared rule; see §6.16) |
| One exact primary architecture selected | **true** |
| One exact fallback architecture selected | **true** |
| Decision justified from the measured frontier | **true** |
| 168-hour positions/games/storage projection produced | **true** |
| Storage constraints analysed | **true** |
| Backend status explicitly reassessed | **true** |
| Parallel evaluation readiness verified | **true** |
| No playing-strength input to the selection | **true** |
| Full suite green | **true** |

21 / 21.

### 6.18 Files, tests and commands

Created:

```text
stratego/training/phase6_soak.py
scripts/run_phase6_agent06.py
scripts/run_phase6_agent06_memory.py        (the §6.16 localization probe)
tests/training/test_phase6_soak.py          (74 tests)
```

Modified:

```text
reports/phase_6_implementation_report.md    (this section only)
```

No existing test was changed or removed. No file under `stratego/engine/`,
`stratego/evaluation/`, `stratego/model/` or `stratego_project_docs/` was touched.

```text
python -m pytest -q                                  ->  2558 passed, 2 skipped, 0 failed   (before)
python scripts/run_phase6_agent06.py                 ->  PASS, 3,922 s
python scripts/run_phase6_agent06_memory.py          ->  the §6.16 localization probe
python -m pytest -q                                  ->  2632 passed, 2 skipped, 0 failed   (after)
```

The 74 added tests cover the soak configuration against the required topology,
the growth and drift statistics, each hard gate wired to the counter it names, the
604,800-second projection, the storage analysis, the knee rule and its robustness,
the strength-exclusion guarantee, the exact architecture records, one real short
soak through the actual pipeline, and the published artifacts themselves.

### 6.19 Data files

```text
reports/phase_6_data/agent_06_soak.json
reports/phase_6_data/agent_06_soak_timeseries.csv          60 samples
reports/phase_6_data/agent_06_weekly_projection.json
reports/phase_6_data/agent_06_architecture_decision.json
reports/phase_6_data/agent_06_memory_localization.json     the §6.16 probe
```

Every headline number in this section also exists in those files.

### 6.20 Handoff to the reviewing chat

```text
Agent 6 status            PASS (21/21 gates)
Phase 6 recommendation    PASS

primary architecture      C1  stratego_transformer_v1 / architecture_family_v1
                              width 128, blocks 4, heads 4, feed-forward 512
                              learned_row_column_v1, pre_layernorm
                              863,959 parameters, model_contract_v2
                              digest 31ca84ab140c...
                              float16 collection @ batch 2,048; float32 evaluation @ single_request
                              10 workers x 1,536 environments, dense legality, snapshot 32

fallback architecture     C0  stratego_transformer_v1 / architecture_family_v1
                              width 64, blocks 2, heads 4, feed-forward 256
                              123,223 parameters, model_contract_v2
                              digest 057d6c9242e3...
                              same precisions, batches and topology

one-hour soak             C1, 3,600.7 s continuous, 19,760 global steps
                          30,351,360 positions, 58,741 games
                          sustained 8,468.7 positions/s, 16.53 games/s, 5.358 GiB/hour
                          0 illegal actions over 30,351,360 legality checks
                          0 frame mismatches, 0 reconstruction mismatches over
                            195,686 verified decisions in 399 games
                          0 worker failures, 0 model/MPS failures
                          0 non-finite outputs over 344,156,160 logits
                          0 swap; drift -0.19%/hour (R2 = 0.002)
                          12/12 soak gates

168-hour compute          5,121,865,011 positions
                          9,996,679 games
                          training ceiling 1.84e9 examples standalone = 0.36 epochs

168-hour storage          900.18 GiB (966.6 GB) raw  -> fits 1 TB external with ~3% spare
                          616.71 GiB (662.2 GB) compressed at Agent 4's measured 0.685
                          does not fit ~150 GB internal at any point
                          checkpoints 557 MiB even hourly

backend                   KEEP_PYTHON remains supported

parallel evaluation       ready; C1 and C0 both load under the strict path and
                          construct greedy and seeded-categorical owners on Metal.
                          Deterministic 1/2/4/8-worker evaluation unchanged.
                          Reference remains float32 + single_request.

full test totals          before 2,558 passed / 2 skipped / 0 failed
                          after  2,632 passed / 2 skipped / 0 failed

artifacts                 reports/phase_6_data/agent_06_soak.json
                          reports/phase_6_data/agent_06_soak_timeseries.csv
                          reports/phase_6_data/agent_06_weekly_projection.json
                          reports/phase_6_data/agent_06_architecture_decision.json
                          reports/phase_6_data/agent_06_memory_localization.json

highest-risk limitation   Host resident memory rose ~191 MiB/hour through the soak
                          and did not decelerate within the hour (device memory was
                          exactly flat, to the byte). The declared gate passed at
                          +0.96% and swap was zero. A follow-up probe localized it:
                          with production recording off the coordinator is flat
                          (+0.8 MiB/hour) and total growth falls to +26.7 MiB/hour,
                          so ~164 MiB/hour belongs to the trajectory-recording path,
                          most likely allocator arena growth from encoding and
                          dropping a ~96 KiB record 16.5 times a second. Bounded in
                          principle, unproven in one hour; naive extrapolation over
                          168 hours is ~31 GiB. Monitor RSS and swap during the final
                          run and restart collection at a checkpoint boundary daily
                          (which resets the arena); settle it with a multi-hour probe
                          against the recording path before relying on an unattended
                          week.
```

No meaningful training occurred. No architecture was invented, no frozen contract
was altered, and Phase 7 has not been started. Only the reviewing chat may
formally accept Phase 6 and freeze the architecture.

---

## 6B. Production Recording Stability Follow-Up

**Status: BLOCKED** — the follow-up's own requirement stands unmet: no
multi-hour persisted-recording soak has yet run to completion. The session
produced three findings, one fixed and regression-tested, one hardened and
awaiting a deterministic replay, one simply requiring the rerun it blocks.

What this section is: the operational validation of the *frozen* Phase 6
configuration — C1, `model_contract_v2`, 10 workers × 1,536 environments, batch
2,048, float16, dense legality, snapshot 32, `KEEP_PYTHON` — with real disk
persistence in place of Phase 6's encode-and-discard. No architecture, contract,
topology, schema or engine decision was reopened, and none changed.

### 6B.1 What was built

Durable trajectory persistence, absent from the repository until now:

```text
stratego/training/shard_writer.py       append-only shard container + verifier
stratego/training/phase6b_recording.py  the persisted-recording soak, the memory
                                        outcome classifier, the machine watchdog
stratego/training/phase6b_recycle.py    process recycling at shard boundaries
scripts/run_phase6b_segment.py          one recycled collection segment
scripts/run_phase6b.py                  the acceptance harness
tests/training/test_phase6b_recording.py
```

The design decisions that matter:

- **Per-worker synchronous writes.** Each of the ten workers compresses and
  writes its own shard files inside the game-sealing call, so the bytes are on
  the filesystem before the call returns. A write backlog is *structurally
  impossible* — `pending_bytes` is identically zero by construction, not
  monitored down to zero — and nothing large ever crosses a pipe. The workers
  idle ~91% of each step, which is where the write latency goes.
- **A container, not a format.** A shard is length-prefixed
  `encode_game_record` payloads — optionally through the repository's existing
  zlib level-6 helper — plus a JSON manifest carrying the record count, byte
  totals, SHA-256 and every game id. `trajectory_v1` is untouched; a record
  extracted from a shard is bit-identical to one the codec produces directly.
  Container overhead is 4 bytes per record plus one preamble and one manifest
  per 128 MiB shard.
- **Crash-ordered.** Data is flushed per record; the manifest is written last,
  at close. A shard without a manifest is an interrupted shard whose complete
  records remain readable; the reader stops cleanly at a truncated tail.
- **Off by default.** `RecordingConfig.output_directory = None` preserves the
  accepted Phase 3/6 encode-and-drop behaviour exactly; every prior test passes
  unmodified.

### 6B.2 The run, and what actually happened

One soak was launched for six hours writing compressed shards to the external
volume. It ran healthily for two and a half:

```text
started            2026-08-12 ~20:14 EDT
aborted            t = 8,981 s (2.49 h), global step ~48,0xx, ~22:43:41 EDT
samples            149 at 60 s cadence
positions          ~74.3 million
sustained          8,273.7 positions/s (settled window; first/last third
                   8,269 vs 8,263 — no drift)
disk               3.546 GiB/hour written, steady to the last sample
compression        0.6773 measured on 8.75 GiB of real persisted games
write errors       0        pending bytes   0 (structural)
swap               27,724,349 bytes at every one of 149 samples — the
                   pre-existing baseline; zero growth attributable to the run
RSS                8.43 → 8.61 GiB across the settled window,
                   +62.9 MiB/hour, R² 0.85 (parsed log series)
```

The abort: the coordinator's per-step sampled-legality check found **slot 112
published as ACTIVE with `legal_count 0`** — an all-zero legality mask — and
raised, exactly as it is designed to. No illegal action reached the engine. The
fault-path shutdown then closed all ten open shards and wrote their manifests
(file mtimes 22:43), so the run ended loud, clean and fully persisted.

**The stall observed from outside was not the soak.** It began *after* the
abort, in the harness's post-soak verification step, whose original
implementation decoded every record of every shard **and retained all of them
in memory simultaneously**. On the dry run's 0.098 GiB that was invisible; on
the real 8.75 GiB corpus it grew without bound: the harness reached 10.8 GiB
RSS with a 566 GB virtual size, drove system swap from 27 MB to **28.8 GiB**,
pushed macOS memory pressure to yellow, and wedged in uninterruptible page-in
waits (`UN`) for ~45 minutes. GPU idleness and the absence of new disk writes
during that window are both explained: the soak had already ended; the wedged
process was a verifier, which uses neither.

The process was killed with the machine at 37% free memory and swap files
consuming internal disk; the system recovered fully within minutes (swap
28.8 GiB → 1.8 GiB, 93% free).

### 6B.3 Post-mortem, item by item

**Last collection progress.** Sample 149: t = 8,940.1 s, global step 48,007,
8,361.8 positions/s, ~22:43 EDT. The abort landed 41 s later, mid-step ~48,0xx.

**Last successful trajectory write.** 22:43 EDT — the shutdown-close of all ten
open shards. 76 shards, 76 manifests: nothing was left unclosed.

**Lead-up series.** Published as
`agent_06b_recording_timeseries.csv` (149 samples parsed from the run log):
throughput flat, disk rate flat at ~3.53 GiB/hour, compression ratio asymptotic
to 0.677, RSS +62.9 MiB/hour, swap bit-identical at every sample, shards
closing on cadence through step 41,900 (the last 128 MiB rollover before the
abort). There is no degradation of any quantity leading up to either failure.
The per-worker RSS split and per-stage write timings were lost with the killed
process — the original harness wrote artifacts only after verification, the
step that wedged. The harness now writes all evidence immediately after the
soak, before any post-processing (§6B.5).

**Cause identification**, against the candidate list:

| Candidate cause | Verdict |
|---|---|
| Retained live objects / true leak | **Yes — in the post-soak verifier**, not in the collection pipeline. Fixed, regression-tested. |
| Allocator fragmentation | Not the stall's cause. In-run growth was +63 MiB/h, swap-flat, for 2.5 h. Its 168-hour behaviour remains **unproven** — explicitly not called bounded. |
| Unbounded trajectory/write queue | No. Writes are synchronous per worker; there is no queue to grow. |
| Filesystem / external-drive backpressure | No. Disk rate steady to the last sample; every record flushed on write; write errors 0. |
| Compression buffering | No. Per-record zlib, no retained buffers, ratio stable at 0.677. |
| The abort itself | A distinct finding: the published zero-legality row (§6B.4). |

**Shard verification.** Every one of the 76 shards was re-read with the fixed
streaming verifier: every record decoded and structurally validated, every
manifest's SHA-256, record count and byte size confirmed, zero duplicate game
ids, zero unclosed shards — **144,149** records, **8.75 GiB** (9.39 GB),
verifier peak RSS **66.7 MiB** against the same corpus that
took the old verifier past 40 GB.

**In-memory data lost at shutdown.** Sealed trajectory records: **none** — the
per-record write-and-flush design means everything sealed was already on disk.
In-flight unsealed games (~1,536, one per environment): never persisted, by
design; a partial trajectory is not a trajectory, and the same loss occurs at
any process boundary including the planned recycling. The full sample dicts
(per-worker RSS, stage timings): lost with the killed harness; the per-minute
log lines survive and are the published timeseries. Run counters: headline
values survive in the log.

### 6B.4 The legality anomaly

One row in ~74.3 million: an active slot published with zero legal actions.
Inspection rules out every cheap explanation — `_evaluate_terminal` runs
unconditionally after every applied action; `has_legal_action` and
`generate_actions_for_player` are equivalent by construction (no
history-dependent exclusions exist in this ruleset); the legality cache dies
with the slot object on every reset; publish derives status, mask and count
from the same state object microseconds apart; and the bulk-synchronous barrier
leaves no concurrent writer. The mechanism is therefore not identifiable from
code inspection, and this report does not guess.

Two things were done instead:

1. **A writer-side trap.** `publish()` now raises at the source if an active
   slot would publish zero legal actions, with the full contradiction in hand:
   game id, generation, ply, acting player, terminal fields, a fresh
   `has_legal_action` evaluation, and the action-history tail. The next
   occurrence is a complete diagnosis instead of a reader-side symptom.
2. **A deterministic reproduction path.** The run is fully determined by
   `(root_seed=60006, sampling_seed, model, topology)`; a replay reaches the
   same state at the same step. That replay is ~1.7 h of saturated machine time
   and is **proposed, not run** — it is a diagnostic soak and stays inside the
   "no further multi-hour runs until the restart test passes" boundary.

Until the anomaly is diagnosed, it is treated as what the evidence says it is:
a once-per-74M-positions correctness event that the pipeline's independent
checking caught before any illegal action reached the engine, and that ends a
production run early. A 168-hour run producing ~5 × 10⁹ positions would expect
~67 such events; **recycling with resume makes each one a segment boundary
instead of a dead run**, which is an operational mitigation, not a diagnosis.

### 6B.5 Fixes landed in this session

1. **Streaming verifier** (`read_shard` / `verify_shard` /
   `directory_summary`): records are decoded, validated and dropped;
   `keep_records=True` exists for tests and small inspections. Digests are
   chunked. Verified live: ~55 MiB peak RSS re-verifying the corpus that took
   the retained-records version past 40 GB. Regression tests pin the
   drop-by-default behaviour.
2. **Machine-level watchdog** in the soak loop: aborts loudly if swap grows
   more than 2 GiB over the run's own baseline or system available memory
   falls below 8% — because a process that drives the machine into swap does
   not fail, it wedges, and the only loud moment is beforehand. Unit-tested,
   including the pre-existing-swap-baseline case.
3. **Evidence-first harness ordering**: the soak JSON and full timeseries CSV
   are written the moment the soak returns, before verification or any other
   post-processing; the files are then updated with verification and gate
   results. The samples that were lost this session cannot be lost again.
4. **Write-failure classification**: a closed-handle write raises `ValueError`,
   not `OSError`; the writer now treats both as the write error they are
   (regression-tested), so a dead handle cannot masquerade as success.
5. **Honest byte accounting**: `ShardStats.bytes_written` now includes shard
   preambles and equals the bytes on disk exactly (regression-tested).
6. **Publish-time zero-legality trap** (§6B.4), converting the anomaly's next
   occurrence into a full diagnosis.

### 6B.6 Restart / resume validation

Run after the fixes above, on the recovered machine, exactly as mandated —
before any further multi-hour soak. Three recycled segments of 1,200 global
steps each (~233 s per segment), C1, the frozen topology, compressed shards to
the external volume, in-worker verification active throughout:

```text
sequence per segment   run -> flush/close shards (manifests written)
                       -> save segment state (JSON)
                       -> orderly worker/coordinator shutdown
                       -> process exit (memory + Metal context released)
                       -> restart: fresh interpreter, candidate rebuilt,
                          configuration digest re-checked
                       -> resume on the next segment's own seed and run id
```

| Verified | Result |
|---|---|
| RSS returns near startup baseline | **yes** — segment baselines 197 / 197 / 196 MiB, drift −0.3% |
| no corrupted trajectory records | **yes** — all 9,655 records across 30 shards decode and validate |
| no missing completed shard | **yes** — 30 on disk vs 30 reported closed |
| no duplicate shard | **yes** — unique names by construction, 0 collisions |
| no duplicated games caused by restart | **yes** — 0 duplicate game ids; segment seeds 60006 / 1060009 / 2060012 |
| run counters remain consistent | **yes** — per-segment totals sum to directory totals exactly |
| configuration/checkpoint identity preserved | **yes** — one configuration digest across all segments |
| elapsed-wall-clock accounting preserved | **yes** — restart overhead (3.04 s total, mean **1.01 s**) is inside the measured wall clock, 0.44% of it |
| collection resumes without manual intervention | **yes** — supervisor launches every segment |
| reconstruction remains exact | **yes** — 22,606 decisions verified in-worker, 0 mismatches |

At the mean measured overhead, even hourly recycling would cost ~0.005% of the
168-hour budget. The interval itself is deliberately **not** frozen here: the
harness derives it from the measured settled slope against a 12 GiB growth
budget (25% of system memory), and the aborted soak's provisional slope of
+63 MiB/hour implies ~194 hours — no restart needed for memory alone if that
slope holds. The binding reason to recycle is the anomaly blast radius (§6B.4)
until it is diagnosed, and segment boundaries are where a 24-hour checkpoint
cadence would land anyway. A full-length recycled soak (§6B.8) settles the
slope properly before any interval is committed.

### 6B.7 Storage, from real persisted bytes — provisional

Measured on the aborted soak's 2.3-hour settled window (stable throughout, but
short of the required 4–6 hours — hence provisional):

```text
written to disk       3.546 GiB/hour compressed   (0.985 MB/s)
produced              5.235 GiB/hour uncompressed
compression ratio     0.6773  (Agent 4's probe predicted 0.685)
container overhead    4 B/record + preamble/manifest per shard  (< 0.1%)
```

Projected to exactly 168 hours at the measured disk rate:

| | GiB | GB |
|---|---:|---:|
| per hour | 3.546 | 3.81 |
| per 24 hours | 85.1 | 91.4 |
| **per 168 hours** | **595.7** | **639.7** |

Against the external volume (931.3 GiB total, to be cleared before the
production run): **64% of capacity, ~336 GiB (~360 GB) of headroom remaining**
after a full week — versus the ~3% margin the raw path offered. Transient
headroom for open shards is 10 × 128 MiB = 1.25 GiB. Every game of the week is
retained; nothing is deleted or tiered.

### 6B.8 What unblocks Phase 6B

In order:

1. The deterministic instrumented replay of the legality anomaly (~1.7 h), now
   that the writer-side trap will catch it with full state context; then
   whatever fix or erratum the diagnosis dictates — noting that if the defect
   proves to live inside the frozen engine, the resolution belongs to the
   reviewing chat, not to this follow-up.
2. A full 4–6 hour persisted-recording soak run as **recycled segments** under
   the supervisor, with the watchdog armed — which simultaneously satisfies the
   duration requirement, exercises restart-in-anger, and bounds both the memory
   question and the anomaly's blast radius.
3. The completion gates as originally specified, including zero swap growth,
   zero write errors, zero backlog, all shards decoding, no duplicate games —
   plus a green full suite before and after.

### 6B.9 Handoff

```text
Phase 6B status            BLOCKED
soak duration              8,981 s continuous (2.49 h of a 6 h target; aborted
                           by the legality anomaly, not by resources)
disk persistence           EXERCISED — 9.2 GiB, 76 shards + 76 manifests, real
                           external-drive writes, per-record flush
compression                EXERCISED — zlib 6, measured ratio 0.6773
sustained positions/s      8,273.7 (settled window, no drift)
sustained disk GiB/hour    3.546 written / 5.235 produced
RSS                        8.43 → 8.61 GiB settled window; +62.9 MiB/hour,
                           R² 0.85; swap growth attributable to the soak: 0
RSS plateaued?             NOT ESTABLISHED — window too short (2.3 h settled),
                           and per instruction this growth is not called
                           bounded without a plateau or tested recycling
recycling required?        YES as operational posture: it bounds both the
                           unresolved memory trend and the anomaly blast radius
tested restart mechanics   PASS — 3 recycled segments, mean 1.01 s overhead,
                           RSS to baseline (−0.3%), 0 duplicate/missing/corrupt
                           records, resume automatic; interval to be derived
                           from a full-length soak's measured slope
168 h storage projection   595.7 GiB (639.7 GB) compressed, provisional
external-drive headroom    ~336 GiB remaining on the cleared 931 GiB volume
full test totals           before 2,632 passed / 2 skipped / 0 failed
                           after  2,697 passed / 3 skipped / 0 failed
                           (+65 Phase 6B tests; the extra skip is the
                           PASS-gated artifact check, armed for a future
                           passing soak)
final recommendation       Phase 6 NOT yet safe to close on the recording
                           path: still BLOCKED pending the anomaly diagnosis
                           and a completed multi-hour recycled soak
```

Artifacts: `agent_06b_recording_soak.json` (BLOCKED, with the full post-mortem),
`agent_06b_recording_timeseries.csv` (149 samples),
`agent_06b_storage_validation.json` (provisional),
`agent_06b_restart_validation.json`. The 8.75 GiB of verified soak shards are
preserved at `/Volumes/Brandon_Washington/stratego_phase6b/soak` pending the
user's disposition. Phase 7 has not been started, and every Phase 6 decision
remains frozen exactly as accepted.

## 6B-2. Phase 6B Continuation — Anomaly Diagnosis and Final Operational Soak

**Status: BLOCKED — frozen engine semantic change appears necessary.** Gate 1's
diagnosis is complete: the legality anomaly is deterministically reproduced,
its root cause is identified with corpus-level corroboration, and the smallest
correct fix lives inside the frozen engine — `create_game` never evaluates the
no-legal-move terminal conditions, so a randomly generated setup can strand the
first player at ply 0 and the engine labels that rules-terminal position
active. Per this continuation's own mandate, a frozen-engine semantic change is
not made unilaterally: the complete reproduction is handed to the reviewing
chat, and Gate 2 (the final 4–6 hour recycled soak) was not started, because
the assignment forbids starting it before Gate 1 passes and because the
measured recurrence rate gives an unfixed six-hour soak roughly a coin-flip
chance of aborting the same way.

Nothing frozen was reopened: C1 remains primary (digest `31ca84ab140c…`,
863,959 parameters), C0 remains fallback (digest `057d6c9242e3…`, 123,223
parameters), and no engine, contract, topology, schema or evaluation semantic
was modified. No source file was changed by this continuation; it added one
diagnostic script and two evidence artifacts.

### 6B-2.1 Starting state, verified

Repository at commit `fb0b6e2` with the previous session's Phase 6B work
uncommitted, exactly as handed off: modified `coordinator.py` (persistence
configuration), modified `worker_pool.py` (shard-writer wiring plus the
writer-side publish trap), new `shard_writer.py`, `phase6b_recording.py`,
`phase6b_recycle.py`, the two run scripts and `test_phase6b_recording.py`.

```text
python -m pytest -q      (before any change of this continuation)
2697 passed, 3 skipped, 0 failed in 159.14s
```

That matches the previous session's recorded ending totals exactly. The
handoff's evidence was verified present rather than assumed: the BLOCKED soak
artifact (149 samples, abort at t = 8,981 s), the restart validation (3
segments, mean overhead 1.01 s, −0.3 % baseline drift, 0 duplicate/missing/
corrupt records), the provisional storage projection (595.7 GiB / 168 h), the
streaming verifier with its drop-by-default regression tests
(`TestVerifierNeverRetainsRecords`), the machine watchdog and its
pre-existing-swap-baseline test, and the preserved 8.75 GiB soak corpus — 76
shards, 76 manifests — on the external volume. The frozen C1/C0 identities
were re-read from `agent_06_architecture_decision.json` and match the handoff
digest for digest.

### 6B-2.2 Gate 1 — the anomaly reproduced, without the 1.7-hour replay

The previous session proposed an instrumented ~1.7 h saturated replay to reach
the failure again. That replay is unnecessary, because the failing state turns
out to be a pure function of slot identity alone. Slot content is derived as
`derive_slot_seed(root_seed, environment_id, generation)` →
`make_random_setups(seed)` → `create_game(...)` — no model, sampler, worker
count or wall-clock input anywhere. The aborted soak ran with `root_seed
60006`; the abort named slot 112. Scanning that slot's generations directly:

```text
(root_seed 60006, environment 112, generation 98)
game_id      batch60006-env000112-gen000098
slot_seed    4213863571973875940
terminal     False        legal actions for acting player   0
acting       red (0)      has_legal_action(red)             False
ply          0            has_legal_action(blue)            True
```

The board makes the mechanism visible — red's front row (row 4, the only row
with anywhere to go at ply 0) is `rB rB r4 r7 rB rB r4 r8 rB rF`:

```text
     a  b  c  d  e  f  g  h  i  j
 10 b4 b6 b9 b7 b3 b8 bB b7 b1 b5
  9 b5 bS bB b5 bB b9 b9 b3 b9 b4
  8 b9 b7 b8 b8 b7 b9 bB b4 b6 b9
  7 b5 b8 bB b6 bB b2 b9 bF b8 b6
  6  .  . ~~ ~~  .  . ~~ ~~  .  .
  5  .  . ~~ ~~  .  . ~~ ~~  .  .
  4 rB rB r4 r7 rB rB r4 r8 rB rF
  3 r5 r6 r6 r8 r9 r5 r7 r8 r7 r7
  2 r9 r9 r9 r6 r4 rS r9 r9 r9 r2
  1 r8 r3 r3 rB r8 r5 r6 r1 r5 r9
```

Every red front-row square on a non-lake column (a, b, e, f, i, j) holds a
Bomb or the Flag — six of red's seven immovable pieces. The four movable
front-row pieces (c, d, g, h) all face lakes. All thirty other red pieces are
boxed in by red's own fully packed rows. Red, the first player, has zero legal
moves at ply 0. Blue has five.

The reproduction was then confirmed at every layer between the engine and the
coordinator (`scripts/reproduce_phase6b_anomaly.py`, ~4 s, exit 0 only when
every stage still reproduces):

| Stage | Result |
|---|---|
| engine `create_game` | `terminal=False`, 0 legal actions, `has_legal_action(red)=False` |
| `BatchSimulator` rebuild through 98 resets | bit-identical state (fingerprint compared), `legal_count 0`, dense mask all-zero |
| real `WorkerPool`, slot 112 advanced to generation 98 | the writer-side publish trap raises: `publish would mark an active slot with zero legal actions: slot 112, game 'batch60006-env000112-gen000098', generation 98, ply 0, acting_player 0, terminal False ('not_terminal'), has_legal_action(acting)=False, battleless 0, last actions []` |
| preserved corpus, all 76 manifests | env-112 generations **0..97 sealed exactly once each, no gaps, no duplicates; generation 98 absent** |
| sum of env-112 sealed game lengths | **48,225** — generation 98 was created and published at global step 48,225 |
| full-horizon scan, 1,536 envs × 120 generations | **exactly one** first-player-stranded setup exists: (112, 98); zero second-player-stranded setups |

The step arithmetic closes the timeline: the soak's last sample was step
48,007 at t = 8,940.1 s; the abort landed 41 s later, and at the measured
5.39 steps/s that is ≈ step 48,227 — the coordinator raised while sampling
from the all-zero mask published at step 48,225. The previous session's
"deterministic, reproducible by replay" claim is confirmed in a strictly
stronger form: the anomaly needs no replay at all.

### 6B-2.3 Root cause

`_evaluate_terminal` — flag capture, then opponent-no-legal-move, then
both-no-legal-move draw, then the draw limits — runs **only inside
`apply_action`** (`stratego/engine/transition.py`). After every applied move it
guarantees the next acting player has at least one legal action, which is why
74.3 million mid-game positions never produced this state. `create_game`
(`stratego/engine/state.py`) performs no terminal evaluation of any kind: it
validates setups, builds the board, sets `acting_player` and returns
`terminal=False` unconditionally. `random_setup` is a uniform shuffle with no
mobility guard. A setup that places Flag/Bomb on all six open front-row
squares therefore produces a game that is already decided under the project
ruleset — `02_project_ruleset.md`: "victory when the opponent has no legal
move; draw if neither player can legally move" — but enters play labelled
active. The engine's own random driver treats that state as an impossibility:
`play_random_game` raises `RuntimeError("non-terminal state with no legal
actions …")` with the comment "an empty list here would mean the terminal
check missed a case". The terminal check misses exactly one case: ply 0.

Every cheaper explanation is excluded by direct evidence:

| Candidate | Verdict |
|---|---|
| legitimately terminal but published active | **YES — the root cause.** Rules-terminal at birth (red cannot move on its turn; blue can → blue wins, `opponent_no_legal_move`), engine reports `not_terminal` |
| wrong legality computation for an active state | No — zero is correct; list, existence check and dense mask agree, and the board confirms it |
| stale metadata after an independent reset | No — fresh `EnvironmentSlot`, correct game id, correct generation; `BatchSimulator` rebuilds the identical fingerprint |
| generation/slot identity crossing | No — manifests show 0..97 sealed exactly once each; 98 is genuinely new |
| shared-memory publication ordering | No — status, mask, count and metadata derive from one state object in one pass; the trap reproduces the contradiction writer-side, before any reader |
| worker/coordinator divergence | No — the coordinator read what the worker published; the worker published what the engine reported |

The frequency closes the quantitative loop. P(six specific setup squares all
immovable, 7 immovables among 40) = 7·6·5·4·3·2 / (40·39·38·37·36·35) =
**1.824 × 10⁻⁶ per game per side — exactly 1 in 548,340**. The aborted soak started
~145,700 games: expected stranded-first-player games 0.266, probability of at
least one ≈ 23 % — unlucky, not anomalous. The horizon scan's one hit in
184,320 (env, generation) pairs (expected 0.336) is consistent. A
second-player-stranded setup (same probability, none in this horizon) is
already handled correctly today: the first player's opening move triggers
`_evaluate_terminal` and the game ends at ply 1.

At production scale the defect is not ignorable, exactly as the handoff
suspected: a 6-hour soak at the measured ~57,900 games/hour expects 0.63
stranded games (**47 % chance of an abort**), and a 168-hour run expects
**~18**.

### 6B-2.4 Why this is BLOCKED rather than fixed

The assignment's Gate 1 protocol distinguishes a pipeline-lifecycle defect
(fix it here) from a frozen-engine semantic defect (stop and report). This one
is unambiguously the latter: the pipeline's publication, reset ordering,
generation identity and legality transport were all verified correct — the
state itself is wrong, and the only place it can be made right is
`create_game`. The two non-engine fixes were considered and rejected:

- **Marking the state terminal in the training layer** would duplicate the
  terminal rule outside the engine (two sources of truth for game semantics, a
  pattern this project has refused at every phase) and would leave the
  engine's own `play_random_game` crashable on the same seeds.
- **Screening or rerolling stranded setups** would change the
  `(root_seed, environment_id, generation) → game` mapping and the collection
  distribution, silently erase a rules-decided game, and break every
  independent rebuild of slot content unless the reroll rule were replicated
  everywhere. It would also fail the assignment's own acceptance test — the
  formerly anomalous state must end up with *correct* status, not cease to
  exist.

Masking options — recycling on the anomaly, weakening the coordinator
assertion, or converting the state to terminal ad hoc at publish time — are
explicitly forbidden by the assignment, and both traps remain in place and
unweakened: the reader-side `verify_sampled_legality` check that aborted the
soak, and the writer-side publish trap that this continuation verified live.

**The proposed fix, not applied**, recorded for the reviewing chat in
`agent_06b_anomaly_diagnosis.json`: after constructing the state,
`create_game` evaluates the initial position's mobility with the same
precedence `_evaluate_terminal` already applies — if the acting player has no
legal action, the game is terminal at ply 0 with `opponent_no_legal_move`
(winner = the mobile opponent) or `both_no_legal_move_draw` (neither side
mobile); flag capture cannot apply at ply 0 and the draw limits are zero
there, so the frozen precedence order is untouched. The blast radius is empty
for every playable game: any game with at least one applied action necessarily
had a mobile first player, so no recorded trajectory, replay, snapshot,
evaluation result or checkpoint changes meaning. The only states affected are
the ones that today crash the pipeline. Downstream composition is already
correct: `publish` reports a terminal slot as `STATUS_TERMINAL`, the
coordinator skips it, auto-reset recycles it one step later, and no
trajectory record is produced for a zero-decision game — the outcome is
counted nowhere today, which the reviewing chat may also wish to address, but
nothing breaks. One implementation note: `state.py` cannot import
`legal_moves`/`transition` at module level (import cycle); a function-level
import inside `create_game` — the pattern `worker_pool.publish` already uses —
or a small `evaluate_initial_terminal` helper in `transition.py` both work.

If the engine is reopened and the fix lands, Gate 1 acceptance is then
mechanical: `scripts/reproduce_phase6b_anomaly.py` must show the trap **not**
firing and (112, 98) terminal with reason `opponent_no_legal_move` and winner
blue; a focused regression pins that game, a stress regression sweeps
constructed stranded setups (both players, including the both-stranded draw),
and Gate 2 — the full 4–6 hour recycled persisted soak — proceeds exactly as
specified.

### 6B-2.5 Gate 2 — not started, deliberately

The assignment is explicit that Gate 2 must not begin until Gate 1 passes, and
Gate 1 cannot pass without the engine decision. Running the soak anyway would
have been a 47 % coin-flip against a known, diagnosed abort — burning six
hours of machine time to re-measure a failure that is already fully
characterized. The Gate 2 machinery itself was verified ready: the recycling
supervisor and segment runner exist and passed their three-segment validation,
the streaming verifier holds ~67 MiB against the 8.75 GiB corpus, the
watchdog is armed, and the persistence path's write/flush/manifest behaviour
was re-confirmed by the corpus forensics this diagnosis performed.

### 6B-2.6 Files, tests and artifacts

Created by this continuation (no source file was modified):

| File | Purpose |
|---|---|
| `scripts/reproduce_phase6b_anomaly.py` | the deterministic five-stage reproduction and diagnosis generator; exits 0 only while every stage reproduces |
| `reports/phase_6_data/agent_06b_anomaly_diagnosis.json` | machine-readable diagnosis: failing identity, all five reproduction stages, candidate-cause verdicts, probability model, proposed fix and rejected alternatives |
| `reports/phase_6_data/agent_06b_final_decision.json` | the BLOCKED decision record and what unblocks Phase 6B |

All previous Phase 6B evidence is preserved unmodified, including the BLOCKED
soak artifacts and the external corpus (used read-only here, exactly for the
forensic purpose it was kept for; it remains at
`/Volumes/Brandon_Washington/stratego_phase6b/soak` pending formal
acceptance).

```text
python -m pytest -q      before this continuation   2697 passed, 3 skipped, 0 failed in 159.14s
python -m pytest -q      after all artifacts        2697 passed, 3 skipped, 0 failed in 155.78s
```

No test was added, removed, weakened or disabled; the three skips are the two
pre-existing Phase 4 capability skips plus the PASS-gated Phase 6B artifact
check, still correctly armed for a future passing soak.

### 6B-2.7 Handoff

```text
Phase 6B status            BLOCKED — frozen engine semantic change appears
                           necessary; reviewing chat decides

starting repository state  commit fb0b6e2 + previous session's uncommitted
                           Phase 6B work; suite 2697 / 3 / 0
ending repository state    identical source; + 1 diagnostic script,
                           + 2 evidence artifacts, + this report section;
                           suite 2697 / 3 / 0

Gate 1
  reproduced?              YES — deterministically, in seconds, at four
                           independent layers; the 1.7 h replay is unnecessary
  root cause               create_game performs no terminal evaluation, so a
                           1-in-548,340 random setup strands the first player
                           at ply 0: rules-terminal, labelled active
  exact failing identity   root_seed 60006, environment 112, generation 98,
                           created at global step 48,225 (abort t = 8,981 s)
  fix                      identified and specified, NOT applied — it changes
                           frozen-engine semantics (stratego/engine/state.py)
  formerly failing run     still fails, by design, until the engine ruling;
                           the reproduction script pins the behaviour
  regressions              reproduction script only; test regressions follow
                           the fix, if authorized

Gate 2                     NOT STARTED (mandated sequencing; 47 % abort risk
                           per 6 h with the engine unfixed; 168 h expects ~18)

correctness counters       illegal actions 0; the one active-with-zero-legal
                           state is fully diagnosed; frame mismatches 0;
                           reconstruction mismatches 0 (22,606 + 144,149
                           records re-verified across the two sessions)

full test totals           before 2697 / 3 / 0 — after 2697 / 3 / 0

Phase 6 recommendation     NOT safe to close on the recording path. Phase 6B
                           remains BLOCKED on one narrow, fully specified
                           engine ruling; everything else — persistence,
                           compression, recycling, streaming verification,
                           storage projection machinery — is validated and
                           waiting.

highest-risk remaining     the stranded-at-birth defect itself: until the
limitation                 engine ruling, every long collection run carries a
                           per-game 1.824e-6 abort probability, and no
                           mitigation short of the semantic fix is
                           permissible under the assignment's rules
```

## 6B-3. Phase 6B Continuation — Authorized Engine Correction and Final Operational Soak

**Status: PASS — both gates.** The reviewing chat authorized the engine
correction diagnosed in §6B-2 as a correctness bug fix to the reference
implementation — ruleset `stratego_project_v1` unchanged, implementation
bumped to `phase2_1_reference_1.2.0` — under ten explicit conditions. This
section records the fix, its differential validation, the zero-decision game
accounting, the Phase 2 revalidation, and the final six-hour recycled
persisted soak, which passed all 26 completion gates and — in the single most
conclusive event of the run — survived, sealed and persisted a live
stranded-at-birth game whose identity had been predicted from seed arithmetic
before launch. The §6B-2 BLOCKED ruling stands as history; this section
supersedes it.

### 6B-3.1 The correction (conditions 1–4)

`_evaluate_terminal`'s mobility rule was refactored into a single shared
implementation, `_evaluate_mobility_terminal(state, next_mover, other)` in
`stratego/engine/transition.py`, used by both call sites so there is exactly
one interpretation of the rule:

- **after every move**, exactly as before: the player about to move is the
  mover's opponent (bit-identical behaviour, proven in §6B-3.3);
- **at game creation**, through the new `evaluate_initial_terminal(state)`
  called by `create_game`: if the first player has no legal action the game
  is terminal at ply 0 — `opponent_no_legal_move` with the mobile opponent as
  winner, or `both_no_legal_move_draw` if neither side can move. If the first
  player can move, creation remains nonterminal even when the second player
  currently cannot; that case is decided at ply 1 by the transition-time
  evaluation, exactly as before. A game decided at creation emits its
  `game_end` event, preserving the exactly-one-per-finished-game contract.

Flag capture cannot apply at ply 0 and the draw counters are zero there, so
the frozen precedence order is untouched. No setup restriction or reroll was
added: a stranding setup is a legal setup that now simply produces a decided
game. The probability stated in §6B-2 was corrected to the exact value:
**1 in 548,340** per game per side (7·6·5·4·3·2 / 40·39·38·37·36·35 =
5,040 / 2,763,633,600 exactly).

Version plumbing: `IMPLEMENTATION_VERSION` moved to
`phase2_1_reference_1.2.0` with the change documented at the constant;
`run_phase5.py`'s frozen-contract pin was retargeted; and the trajectory
validator accepts the declared compatibility set {1.1.0, 1.2.0} — records
written under 1.1.0 replay and reconstruct identically under 1.2.0, which
§6B-3.3 proves on the real corpus rather than asserts. Both zero-legality
traps — the coordinator's `verify_sampled_legality` and the worker's
publish-time trap — remain in place and unweakened (condition 10).

### 6B-3.2 Zero-decision games are full citizens (condition 7)

A game terminal at creation ("stillborn") previously would have vanished: no
decision is ever recorded for it, so no builder existed, no outcome was
written, and the reset erased it. `trajectory_v1` turned out to represent the
empty game **without any schema change** — the wire format already permits
zero decisions and actions, and the only gap was that the builder took its
ply-0 snapshot lazily on the first decision. Two collection-layer changes
closed this:

- `GameTrajectoryBuilder.finish` takes the ply-0 snapshot itself for a
  zero-decision game (the only path that can reach `finish` with no
  decisions — any other premature seal fails the action-history check);
- the worker seals a stillborn game at its reset boundary: `record_outcome`
  feeds `episode_count` and the `last_*` fields exactly like any completed
  game — so `collect_finished`, `games_finished` and the terminal-reason
  tallies include it — and when recording is enabled its zero-decision record
  is sealed through the ordinary `finalise_recording` path, persisted with
  its setups, winner and terminal reason. A `total_stillborn_games` counter
  flows through the worker replies, the recording totals and the segment
  state files, so the count is reconciled against the persisted records
  rather than trusted.

The lifecycle preserves the coordinator's one-game-per-slot-per-phase
accounting exactly: a stillborn created by a reset at step S is published
terminal at S, skipped by the coordinator, and sealed-then-recycled at step
S+1.

### 6B-3.3 Differential validation of the corrected engine (condition 8)

Captured with the 1.1.0 engine before any edit, re-run identically after:

| Baseline | Cases | Differences |
|---|---:|---|
| initial-state fingerprints, all 1,536 × 120 slot identities of the aborted soak's horizon | 184,320 | **exactly one** — (112, 98): `not_terminal` → `opponent_no_legal_move`, winner blue |
| complete `play_random_game` games, seeds 0–1999, full-history final fingerprints | 2,000 | **0** |

The stored corpus — all 76 shards of the aborted soak, recorded under
1.1.0 — was then verified under the corrected engine:

- **full streaming decode + structural validation**: 144,149 records, 0
  decode failures, 0 validation failures, 0 duplicate game ids, all digests
  and manifests intact;
- **dense sample reconstruction** (every 50th record of every shard plus all
  98 env-112 records): **3,017 games** rebuilt from their stored setups with
  the new engine, **1,589,012 actions replayed**, **51,167 stored snapshots**
  compared fingerprint-for-fingerprint against the replayed states, terminal
  outcome and final ply checked per game — **0 mismatches**. Snapshot 0 of
  every sampled game also matches a fresh `create_game` from the stored
  setups, closing the loop between storage and the corrected creation path.

The anti-leak workload counters that differ between today's runs and the
accepted Phase 2 metrics file (86,500 vs 103,625 valid trials) were traced to
accepted-era harness parameters, not the engine: today's anti-leak stage run
against the **old** engine in a clean worktree produces counters and
mismatches (all zero) byte-identical to the new engine's.

### 6B-3.4 Phase 2 acceptance suite, rerun in full (condition 9)

`run_phase2_validation.py` at full scale, written to
`reports/phase_2_data/phase_2_metrics_reference_1_2_0.json` — the accepted
1.1.0 metrics file is preserved untouched:

```text
replay reconstruction     10,000 games, 5,078,406 plies — 0 state, 0 event,
                          0 observation, 0 result mismatches
random-game tallies       identical to the accepted run: 2,831 red wins,
                          2,846 blue wins, 4,323 draws, longest 1,860 moves
anti-leak                 0 mismatches in every category
combat / legality /       120/120 combat cases; 9,285 legality positions,
invariants / mirror       0 mismatches; 1,045,111 invariant transitions,
                          0 violations; 1,804 mirror pairs, 0 mismatches
embedded full pytest      2,722 passed, 3 skipped, 0 failed
harness's own verdict     unexplained mismatches across every gate: 0
```

### 6B-3.5 The permanent regressions (condition 6)

35 new tests across three files:

- `tests/engine/test_initial_mobility.py` (12): constructed red-stranded /
  blue-mobile → blue win at ply 0 with one `game_end` event; both stranded →
  draw at ply 0; red mobile / blue stranded → nonterminal at creation and
  decided at ply 1 as before; ordinary mobile positions unchanged across
  four seeds; snapshot and replay round-trips of born-terminal states.
- `tests/training/test_stillborn_games.py` (13): the exact
  `(60006, 112, 98)` production case through `BatchSimulator` (terminal,
  reason, winner, empty legality products, outcome, identity); the
  reset-through-a-stillborn-generation lifecycle; terminal-slot step
  refusal; the zero-decision `trajectory_v1` round-trip, validation and
  reconstruction; and a real `WorkerPool` publishing the stillborn slot as
  TERMINAL, sealing outcome and record on the first step, and persisting a
  decodable zero-decision shard record. Fixture seeds (157345, 151139,
  1032652) were located by exhaustive scan and are pinned forever.
- `tests/training/test_phase6b_recording.py` (+10): the Gate 1
  fix-validation artifact must show every stage correct and the exact
  548,340 reciprocal; the Gate 2 artifacts must be internally consistent,
  recycled, gate-complete on PASS, and stillborn-reconciled. All ten execute
  rather than skip, because the artifacts exist before the final suite runs.

Gate 1 acceptance is `scripts/reproduce_phase6b_anomaly.py`, repurposed: the
exact deterministic sequence that aborted the first soak now runs to the
correct result at every layer — engine (terminal at creation, blue win, one
`game_end`), batch (bit-identical), production pool (slot 112 / generation 98
published `STATUS_TERMINAL`, no trap, outcome sealed, slot recycled to
generation 99), preserved corpus undisturbed, horizon scan unchanged. The
pre-fix diagnosis artifact is frozen as history;
`agent_06b_anomaly_fix_validation.json` records the passing rerun.

### 6B-3.6 Gate 2 — the final recycled soak

Run as `scripts/run_phase6b_final.py`: unlike the first attempt's continuous
soak, the 6-hour logical run **is itself recycled** — five segments of 72
minutes, each a fresh child process on its own root seed (base 70,007, a new
family so no game identity can collide with either preserved corpus) and its
own run id, with flush/close-manifests, state persistence, orderly shutdown,
process exit, restart, configuration-digest recheck and automatic resume at
every boundary. Restart time is wall clock spent from the budget.

```text
segments                   5 × 4,320 s        wall clock 21,607.8 s (6.002 h)
restart overhead           5.44 s total       0.025 % of wall; 1.0–1.2 s each
positions                  179,885,567        decisions recorded: identical
games                      349,685            records persisted:  identical
sustained (settled)        8,334.8 pos/s      mean game 511.6 plies
disk (settled)             3.572 GiB/h        compression ratio 0.6773
disk (whole-run wall)      3.510 GiB/h        includes warmups and restarts
shards                     200 + 200 manifests, 21.07 GiB, zlib level 6
in-worker verification     1,153,116 decisions reverified, 0 mismatches
```

**The stillborn event.** The pre-launch scan predicted exactly one
first-player-stranded identity inside the reachable horizon:
`batch3070016-env000259-gen000037` (segment 3, environment 259, generation
37). Roughly an hour into segment 3 that game was created live, and the run
did what §6B-3.1 and §6B-3.2 say it must: published the slot as TERMINAL —
no trap, no anomaly — sealed the outcome (blue win, `opponent_no_legal_move`),
persisted a zero-decision record (1 ply-0 snapshot, validates clean, shard
`p6bf601493g003_w01_s000002` index 858), recycled the slot, and continued to
the end. Worker-counted stillborn games: 1; persisted zero-decision records:
1; the same game. Under `phase2_1_reference_1.1.0` this exact moment aborted
the production soak at t = 8,981 s. The fix is validated in production, not
only in tests. (The same horizon held one *second*-player-stranded setup —
environment 517, generation 55 of the same segment — which is the
already-handled case: a 1-ply red win by the transition-time evaluation.)

**Memory (§15 acceptance rule).** No natural plateau is claimed; recycling
demonstrably bounds memory instead:

```text
fresh-process baselines    196 / 197 / 196 / 196 / 197 MiB   drift +0.33 %
within-segment RSS slopes  +192.6 / +121.1 / +105.4 / +164.7 / +85.4 MiB/h
                           (settled windows, R² 0.89–0.92, total RSS
                           6.75–7.07 GiB across coordinator + ten workers)
swap growth                0 bytes — the system-wide maximum over all 360
                           samples equals the pre-run baseline exactly
watchdog                   armed throughout, never tripped
```

Every boundary returned the next process to the startup baseline; no
cumulative drift exists after five fresh starts. **Recommended operational
recycle interval: 12 hours**, derived rather than chosen: the 12 GiB growth
budget (25 % of system memory) divided by the worst observed settled slope
(192.6 MiB/h) is 63.8 h; halved for the Phase 7 learner/optimizer memory not
yet present (~32 h); halved again as operating margin and rounded to a
checkpoint-aligned cadence. At that cadence a 168-hour run performs 14
restarts costing ~15 s total (0.0002 % of the budget).

**Streaming verification.** The whole final corpus, decoded and structurally
validated in a dedicated subprocess so its memory is its own:

```text
records verified           349,685 of 349,685 (the entire corpus)
decode / validation fails  0            duplicate game ids   0
unclosed shards            0            manifest mismatches  0
verifier peak RSS          110.0 MiB    (vs 8.75 GiB corpus → 40 GB+ under
                                        the retired pre-6B verifier)
verifier swap caused       0            duration 3,029 s
```

**Storage, measured again (condition: do not reuse 0.6773 unmeasured — it
was remeasured and reproduced).** Projected to 168 hours:

| basis | GiB/h | 168-hour GiB | volume headroom |
|---|---:|---:|---:|
| whole-run wall (incl. warmups + restarts) | 3.510 | **589.7** | 341.5 GiB |
| settled windows | 3.572 | 600.2 | 331.1 GiB |

Against the 931.3 GiB external volume (cleared before production), a full
week fits with roughly a third of the disk to spare; transient headroom for
open shards is 10 × 128 MiB, and manifests/state files measure in megabytes.
Preserving every game of the week remains the policy.

**Hard failure conditions (§18):** all clean — 0 illegal actions, 0
active-with-zero-legal anomalies, 0 frame mismatches, 0 reconstruction
mismatches, 0 unhandled worker failures, 0 model/MPS failures, 0 non-finite
outputs, 0 corrupt/missing/duplicate records, 0 write errors, backlog
structurally zero, swap growth zero, baselines flat, resume automatic in all
five launches. 26 / 26 completion gates true; recommendation **PASS**.

### 6B-3.7 Files, tests and artifacts

Source changes (the authorized correction and its accounting):

| File | Change |
|---|---|
| `stratego/engine/transition.py` | shared `_evaluate_mobility_terminal`, new `evaluate_initial_terminal` |
| `stratego/engine/state.py` | `create_game` evaluates initial mobility |
| `stratego/engine/constants.py` | `IMPLEMENTATION_VERSION` → `phase2_1_reference_1.2.0` |
| `stratego/training/trajectory.py` | implementation-version compatibility set; ply-0 snapshot for zero-decision `finish` |
| `stratego/training/worker_pool.py` | stillborn sealing at reset boundaries; `total_stillborn_games` counter |
| `scripts/run_phase5.py` | frozen-contract pin retargeted to 1.2.0 |
| `scripts/run_phase6b_segment.py` | reports `stillborn_games` in segment state |
| `stratego/evaluation/__init__.py` | docstring version reference |

Created: `tests/engine/test_initial_mobility.py`,
`tests/training/test_stillborn_games.py`, `scripts/run_phase6b_final.py`
(the recycled-soak harness), and the artifacts
`agent_06b_anomaly_fix_validation.json`, `agent_06b_final_soak.json`,
`agent_06b_final_soak_timeseries.csv` (360 samples),
`agent_06b_recycling_validation.json`,
`agent_06b_final_storage_validation.json`,
`reports/phase_2_data/phase_2_metrics_reference_1_2_0.json`, and the updated
`agent_06b_final_decision.json`. All §6B-2 evidence — the BLOCKED soak
artifacts, the frozen pre-fix diagnosis, and both earlier corpora — is
preserved unmodified.

```text
python -m pytest -q    before this continuation       2697 passed, 3 skipped, 0 failed
python -m pytest -q    after Gate 1, before Gate 2    2726 passed, 9 skipped, 0 failed
python -m pytest -q    final                          2732 passed, 3 skipped, 0 failed
```

The final three skips are the two pre-existing Phase 4 capability skips plus
the first soak artifact's BLOCKED-branch skip, which is correct: that
artifact is preserved history. No test was removed, weakened or disabled.

### 6B-3.8 Handoff

```text
Phase 6B status            PASS — both gates

Gate 1                     anomaly deterministically reproduced (seconds, four
                           layers); root cause create_game's missing initial
                           mobility evaluation; fixed as authorized in
                           phase2_1_reference_1.2.0 via one shared rule
                           implementation; formerly failing sequence passes;
                           35 permanent regressions; differential validation
                           total (184,320 + 2,000 + 144,149-record corpus +
                           full Phase 2 suite: only (112,98) differs, by
                           design)

Gate 2                     6.002 h recycled soak, 5 segments, 5.44 s restart
                           overhead (0.025 %); 179,885,567 positions;
                           349,685 games = 349,685 persisted records;
                           8,334.8 positions/s sustained; 3.572 GiB/h
                           settled disk rate at ratio 0.6773; one predicted
                           stillborn game encountered live, sealed, persisted
                           and reconciled; 26/26 gates; PASS

memory                     baselines 196/197/196/196/197 MiB (+0.33 %); slopes
                           +85 to +193 MiB/h within segments, fully reset at
                           every boundary; swap growth 0; recommended
                           production recycle interval 12 h (derived)

persistence                200 shards + 200 manifests, 21.07 GiB, 0 write
                           errors, 0 corrupt/missing/duplicate records

streaming verifier         349,685/349,685 records, 0 mismatches, 110 MiB
                           peak RSS, 0 swap

168-hour storage           589.7 GiB (wall basis) / 600.2 GiB (settled) of
                           931.3 GiB — ~341 GiB headroom; fits with margin

correctness                0 illegal actions, 0 active-with-zero-legal, 0
                           frame mismatches, 0 reconstruction mismatches
                           (1,153,116 decisions reverified), 0 worker/MPS
                           failures, 0 non-finite outputs

full test totals           2697/3/0 → 2732/3/0 (+35, none removed/weakened)

Phase 6 recommendation     SAFE TO FORMALLY CLOSE on the recording path

highest-risk limitation    the 6-hour soak bounds but does not directly
                           observe 168-hour behaviour: production depends on
                           the recycling supervisor operating at ≤ 12 h
                           cadence, and the storage projection extrapolates a
                           6-hour rate 28-fold with ~341 GiB of headroom
                           absorbing drift
```
