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
