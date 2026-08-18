# Phase 9 Implementation Report

Population self-play reinforcement learning: freeze the RL contract, collect
on-policy rollouts from an immutable behavior snapshot against a mixed
population (current / historical / rule / stress), train C1 with PPO under
behavior-KL damping and continued belief supervision, and prove direct
self-improvement over the accepted Phase 8 anchor under predeclared gates.

Phase 9 executes against the frozen post-Phase-8 stack: rules
`stratego_project_v1`, reference engine `phase2_1_reference_1.2.0`,
observation `observation_v2_1_127ch`, engine action encoding
`source_destination_10000_v1`, model contract `model_contract_v2` (C1,
863,959 parameters, config digest `31ca84ab…`), backend `KEEP_PYTHON`,
trajectory `trajectory_v1` (snapshot interval 32), the frozen
`setup_library_v1` / `setup_sampler_v1` / `setup_source_v1` stack with the
`neutral_v1` profile, and the accepted Phase 8 checkpoint
`checkpoints/phase8/warmstart_c1_v1.pt`
(`f7e9c40d0f160da00176596755c20768ba32561a26f9178dbb4a95e889eec7ca`, update
24,000). Phase 9 is **not** the official 168-hour campaign: no decision-time
search, no MCTS, no learned setup selection, no human data, no architecture
change, and no rule change is authorized anywhere below. Formal Phase 9
acceptance remains with the reviewing chat after Agent 8 reports.

## 1. Agent 1 — RL Contract, Evaluation Banks, and Acceptance Freeze

**Status: PASS** — 18 / 18 completion gates true. Machine-readable record:
`reports/phase_9_data/agent_01_rl_contract.json` (the full frozen contract +
digest), `agent_01_acceptance.json` (verification, anchor baseline, gates,
handoff), `agent_01_validation_bank.json` and `agent_01_test_bank.json` (the
two frozen banks with manifests and structural audits). Every value below was
frozen while **no Phase 9 optimizer step had run and no trainable Phase 9
rollout existed** (`data/phase9/rollouts/` absent, zero learner checkpoints,
recorded in the artifact). Acceptance command:
`python scripts/run_phase9_agent01.py --run-pytest` (stages `verify`,
`banks`, `export`, `anchor`, `artifacts`; each stage is independently
resumable).

### 1.1 Prerequisites verified

Phase 8 formal acceptance re-verified from
`agent_07_final_acceptance.json` (status PASS, 42/42 gates) and the Phase 8
report. Both frozen checkpoints verified by SHA-256 **and** the normal load
path (`load_model_for_evaluation`, CPU): accepted
`warmstart_c1_v1.pt` = `f7e9c40d…ec7ca`, 863,959 parameters, global step
24,000; canonical untrained `warmstart_c1_v1_initialisation.pt` =
`01c907ee…4754c` with model-state checksum `cfe60bb0…042b8`. C1 config digest
`31ca84ab…fe07d` re-derived live. The accepted corpus resolves only through
`synthetic_corpus.default_corpus_root()` to the accepted location, and all
three accepted digests (content `c95c3545…`, metadata `1db0f02f…`,
commit-index `32e8e18d…`) verify at identity level (Agent 1 consumes no
corpus payloads). Phase 4 roster (4 ladder + 6 stress), Phase 7 library
digest `7b8a6660…`, Phase 4 bank digest, sampler/profile constants, and
trajectory foundation all match frozen expectations (`verify_phase9_upstream`
returns no problems; suite baseline before any edit: 3,797 passed / 3
skipped in 243 s at commit `0fe6caf`).

### 1.2 The nine frozen contracts

`stratego/training/phase9_seed.py` + `stratego/training/phase9_contract.py`
freeze, serialize and regression-test every learning-design decision.
Contract document digest (canonical JSON, SHA-256):
**`ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34`**.

- **Seeds** (`strat-rl9` blake2b personalization, domain-separated, no
  global RNG cursor): master 2026081601, rollout schedule 2026081602
  (reserved — the schedule is pure arithmetic), opponent schedule
  2026081603, training order 2026081604, pilot namespace 2026081605,
  canonical namespace 2026081606, validation bootstrap 2026081607,
  final-test bootstrap 2026081608.
- **`phase9_rollout_v1` game identity**:
  `phase9_rollout_v1|ms=…|ns=<namespace>|it=NNN|b=<bucket>|g=NNNN` over
  namespaces `canonical` / `pilot_p9a..p9f`, 1-based iterations, buckets
  `current|historical|rule|stress`. Per-game streams: `setup_root`,
  `opponent:historical`, `policy:red`, `policy:blue`, per-decision
  `behavior_sampler`. Rule/stress per-ply randomness stays on the frozen
  Phase 4 `derive_decision_seed` path.
- **`phase9_population_v1`**: canonical 2,048 = 1,024 current + 512
  historical + 307 rule (154 Strategic / 107 Tactical / 46 Basic, contiguous
  ordinal subranges) + 205 stress (`(ordinal + iteration) % 6` rotation over
  the frozen six-policy roster — exact counts, balanced over six
  iterations); pilot 1,024 = 512/256/154 (77/54/23)/102. Colour balance:
  learner is red iff `(ordinal + iteration) % 2 == 0`; odd remainders
  alternate by iteration parity. `learner_control ∈ {red, blue, both}`;
  training eligibility: current = both colours, all asymmetric buckets =
  current-policy side only; opponent decisions stay in the trajectory for
  reconstruction but receive no Phase 9 loss.
- **Historical league**: `H000` = the accepted Phase 8 checkpoint; archive
  `H%03d` after every 5th committed iteration; active window = anchor + 8
  most recent snapshots created strictly before the current iteration;
  uniform draw via the frozen opponent stream; archives immutable.
- **Behavior policy and `phase9_rollout_store_v1` storage decision**:
  behaviour = legal softmax at temperature 1.0 from the frozen behavior
  snapshot; rollout action selection = ascending-legal-order inverse CDF
  against `behavior_sample_uniform(game_id, ply)`; evaluation stays greedy.
  **`trajectory_v1` is reused faithfully** — `DecisionRecord.
  old_probabilities` stores the full legal behavior distribution (float32,
  its Phase 3 meaning), `collection_policy_version` carries the acting
  side's policy token (its documented checkpoint-identifier slot), and
  `GameRecord.collection_checkpoint_id` carries the behavior snapshot
  SHA-256. Everything trajectory_v1 has no field for (bucket, learner
  control, opponent identity/digests, seeds, setup provenance) lives in the
  `phase9_rollout_store_v1` metadata sidecar + commit journal (the accepted
  Phase 8 corpus-commit shape) under `data/phase9/rollouts/`. Training-time
  `pi_b(a_t|s_t)` is the **stored** float32 probability
  (`log pi_b = ln(max(p, 1e-12))`); verification recomputes under the exact
  snapshot (CPU float32 reference) with frozen tolerance
  **max |p_stored − p_recomputed| ≤ 1e-4** per legal entry; a snapshot-digest
  mismatch is a hard veto, never a tolerance question. Rule/stress opponent
  decisions store the accepted Phase 8 one-hot representation with the
  policy token and neutral value prediction and are never PPO-trainable.
  Rollout lifecycle `COLLECTING → SEALED → TRAINING → EVALUATED → COMMITTED`
  with the six frozen crash rules (regenerate only missing committed IDs;
  sealed rollouts immutable; exact logical resume; no cross-iteration
  generation; one behavior identity per iteration).
- **`phase9_advantage_v1`**: per-game, per-learner-colour sequences of that
  player's own decisions; `v_t = P_t(W) − P_t(L)` from the stored behavior
  W/D/L; `γ = 1.0`, `λ_A = 0.5`, `λ_V = 0.8`; `δ_t = v_{t+1} − v_t`, or
  `z − v_t` at termination (`z ∈ {−1, 0, +1}`); `A_t = δ_t + λ_A A_{t+1}`;
  W/D/L lambda targets `Y_t = Z` at the terminal decision else
  `(1−λ_V) P_{t+1} + λ_V Y_{t+1}` (soft categorical CE). Filter
  `τ = max(Q_0.75(|A|), 0.01)` per sealed iteration (linear-interpolation
  quantile), PPO-eligible iff `|A_t| ≥ τ`; standardization over the
  PPO-selected subset only (population std, ε = 1e-8); value/belief train
  on every learner decision.
- **PPO and damping**: ratio against the stored behavior probability, clip
  ε = 0.20; `D_KL(π_b ‖ π_θ)` over every learner decision of the minibatch,
  target 0.015, β update once per epoch (×2 above 0.0300, ×0.5 below
  0.0075, clamp [1e-4, 0.2]); hard limits KL > 0.08 and clip fraction >
  0.75 FAIL/VETO. Full loss `L = L_PPO + 0.5·L_value + 0.25·L_belief +
  β·KL − c_H·H(π)`; entropy coefficient linear 0.005 → 0.001 over the run's
  own iteration budget. `phase9_train_order_v1`: per-epoch seeded shuffle of
  the sealed learner-decision universe, contiguous 512-decision minibatches,
  final partial minibatch consumed. Optimizer: float32, MPS, AdamW
  (0.9/0.999, ε 1e-8), weight decay 0.01, grad clip 1.0, 2 epochs/rollout,
  constant LR; LR and initial KL β come only from the pilot matrix.
- **`phase9_checkpoint_v1`**: the 27 frozen required fields (model /
  optimizer / scheduler state, step + iteration + minibatch cursor +
  examples, behavior + rollout identities and digests, KL controller +
  entropy position, population + historical identities/digests, schedule +
  sampler versions, best-validation records, all seeds, corpus + stack
  versions, wall-clock and runtime versions); paths are diagnostic only.
- **Pilot matrix** (exactly six, no seventh run, no early stop): P9-A..P9-F
  = LR {1e-4, 3e-4, 6e-4} × initial KL β {0.005, 0.020}; each fresh from
  the Phase 8 checkpoint, 8 iterations × 1,024 games × 2 epochs; validation
  passes after iterations 4 and 8; selection = frozen validation score at
  iteration 8 with the frozen tie-break; 12 hard vetoes (all
  zero-tolerance counters, KL/clip limits, validation Random ≥ 0.90 and
  Basic ≥ 0.60); final-test results are forbidden evidence.
- **Canonical run**: 60 iterations × 2,048 games (122,880 max), 2 epochs,
  validation + archive every 5 iterations, 12-hour operational ceiling
  (never permission to shorten the logical contract silently); best
  checkpoint = strictly highest frozen validation score.
- **`phase9_acceptance_v1` final gates** (Agent 8, sealed test bank):
  A anchor EWR ≥ 0.58 with paired 95% LB > 0.53 (512 pairs / 1,024 games);
  B/C Strategic/Tactical final EWR ≥ 0.52 **and** paired improvement over
  the anchor ≥ +0.05 with CI LB > 0 (stretch report-only 0.55); D Random
  ≥ 0.94 overall, ≥ 0.90 per colour, paired LB > 0.92; E Basic ≥ 0.65,
  LB > 0.60; F zero illegal/failures/non-finite/observer violations;
  G collapse fraction (max legal probability > 0.999) < 0.25; H belief
  retention on the accepted Phase 8 held-out benchmark (CE ratio ≤ 0.98,
  top-1 above the remaining-count prior; teacher-imitation CE report-only).
  The Agent 8 anchor procedure is frozen now: the anchor replays the same
  final cases vs Tactical/Strategic and the improvement CI is the frozen
  paired-difference bootstrap. Statistics everywhere:
  `paired_unit_percentile_bootstrap`, numpy PCG64, 10,000 resamples, 95%,
  interval seed = `matchup_seed(bank bootstrap seed, matchup token)`.
  Report-only diagnostics (21 frozen) may not rescue a failed hard gate.

### 1.3 Evaluation banks

`stratego/evaluation/phase9_banks.py` builds both banks deterministically
from frozen constants through the **untouched** `setup_sampler_v1`
(`neutral_v1` profile): case identity is family-major
(`setup_pair_id = family_index × cases_per_family + ordinal`), both sides of
a case draw from the case's family (family purity — the only
colour-symmetric choice under `color_swap_same_board`), and each side is the
first attempt `k = 0, 1, …` of `sample_setup(split,
eval_bank_draw_seed(bank, family, case, side, k))` whose primary family
matches (deterministic rejection; ceiling 2,048; observed maxima 101 /
116). Construction plays no game and reads no strength signal.

```text
phase9_validation_bank_v1   128 cases  8 × 16 families   validation split
  bank digest      3d28d544f6669129b12c13e4e3738aa36d1a99e4af8f6685bbb032793701ee4a
  manifest digest  fe0bae7fec38b073da99c6e4125d429c1b1d803922481458bca8a1bff632eca5
phase9_test_bank_v1         512 cases  32 × 16 families  test split
  bank digest      f38e405559fc7c04b0832b1d3a4e3d82cd68ffff29bc1a9af456a3940e1de6a7
  manifest digest  43a88f2c96529727e4d04251f458a26eaf88329dd819ed45c61d488a006eda0d
```

Structural audits pass in full for both banks: exact counts and family
balance, family purity, split isolation (every base index re-derived into
the required split), engine validity of every pair, provenance rebuilds
through `rebuild_from_provenance`, isolated-rebuild spot checks, distinct
positions, zero overlap between the two banks, digest stability across
independent rebuilds. Core opponents: Phase 8 anchor + Random / Basic /
Tactical / Strategic; stress schedules are report-only bank prefixes (32
pairs per stress policy on validation, 64 on final test). Validation score
`S = 0.45·E_Strategic + 0.35·E_Tactical + 0.20·E_Phase8-anchor` with the
frozen four-step tie-break; Random and Basic are regression guards, not
score components.

**Sealing** (pure, stateless, regression-tested):
`check_test_bank_access` allows `structural_audit` to every agent, seals
`final_evaluation` until Agent 8, and refuses
`neural_model_inference` / `model_metric` / `checkpoint_selection` /
`hyperparameter_selection` outright before Agent 8;
`check_validation_bank_access` allows the model-selection purposes to every
agent and refuses `weight_update` always. No neural model touched the
final-test bank in this agent (measured: zero test-bank games; every anchor
schedule pinned to the validation bank version).

### 1.4 Phase 8 anchor baseline on the validation bank

Explicitly permitted before the first update (the validation bank exists
for model selection). The anchor played through a bitwise-verified
evaluation export (`load_model_for_evaluation` → frozen Phase 6
`save_checkpoint`; state-dict equality proven) as
`phase6_c1_warmstart_greedy@0.2.0+float32`, greedy, float32, MPS owner,
8 pure-engine workers, all 128 pairs per opponent (256 games each, 1,024
total, 476 s ≈ 2.15 games/s), intervals =
`matchup_seed(2026081607, matchup)`:

```text
random_legal            EWR 0.9531   95% [0.9277, 0.9766]   W/D/L 243/2/11
basic_heuristic         EWR 0.5957   95% [0.5430, 0.6484]   W/D/L 151/3/102
tactical_rule_based     EWR 0.4551   95% [0.3984, 0.5098]   W/D/L 115/3/138
strategic_rule_based    EWR 0.4531   95% [0.4043, 0.5020]   W/D/L 115/2/139
```

Safety: zero illegal actions, zero policy errors, zero inference failures,
zero worker torch imports, zero worker checkpoint loads.

**Risk notes (recorded in the artifact, not mine to relax):** the anchor's
Basic EWR (0.5957) starts *below* the frozen pilot Basic veto floor (0.60)
and final gate E (0.65) on this bank — a run that fails to improve against
Basic sits at or under the veto line, which is the deliberate design: the
frozen thresholds demand genuine improvement. Tactical/Strategic baselines
(0.4551 / 0.4531) confirm the Phase 8 finding that the imitation warm start
sits below even against the strong tiers; gates B/C require +0.05 paired
improvement with CI LB > 0.

### 1.5 Tests and completion gates

New regression files: `tests/training/test_phase9_seed.py` (23),
`tests/training/test_phase9_contract.py` (72),
`tests/evaluation/test_phase9_banks.py` (18, pins both frozen bank
digests), `tests/training/test_phase9_agent01_artifacts.py` (18,
artifact-gated). Every required control is covered: contract round-trip,
seed domain separation (including disjointness from every Phase 8 stream),
exact game-count / proportion / rule-subdivision arithmetic,
learner-control and colour-balance semantics, hand-computed
advantage/δ/W-D-L-lambda sequences, filter threshold and quantile rule,
pilot matrix exactly six, validation-score and final-gate arithmetic,
test-bank refusal and validation-bank allowance, bank family balance and
colour-pairing exactness, and zero final-test model inference. Suite:
3,797 passed / 3 skipped before; **3,928 passed / 3 skipped** steady state
after (the in-run record inside the acceptance artifact shows 3,910 + 21
skipped because the 18 artifact-gated tests skip while the artifacts are
being produced; they pass once the freeze exists).

All 18 completion gates true: `phase8_identity_verified`,
`corpus_resolver_verified`, `corpus_digests_match`, `rl_contract_frozen`,
`population_contract_frozen`, `rollout_schedule_frozen`,
`behavior_storage_semantics_frozen`, `advantage_contract_frozen`,
`checkpoint_contract_frozen`, `pilot_matrix_exactly_six`,
`validation_score_frozen`, `validation_bank_frozen_and_hashed`,
`test_bank_frozen_and_hashed`, `test_bank_neural_access_zero`,
`final_gates_frozen`, `no_phase9_optimizer_steps`,
`no_trainable_phase9_rollouts`, `full_suite_green`.

### 1.6 Handoff to Agent 2

Everything Agent 2 needs is in `agent_01_acceptance.json →
handoff_to_agent_2`: the nine contract identities + digest, exact bucket
schedules, the game-ID / opponent-ID specifications, the colour-balance
rule, the historical-archive identities and active-window rule, the setup
assignment rule (`training_setup_source('neutral_v1')`, environment 0,
generation 0, root = `setup_root_seed(game_id)`), both bank manifest
digests, all seed derivations, and learner-control semantics. Agent 2 makes
no new learning-design decision. Operational note: the cleared external
drive is available for Phase 9 rollout storage; the store contract records
location in manifests while identity remains version + digests (the
accepted Phase 8 relocation precedent), so using it changes nothing
logical.

## 2. Agent 2 — Population and Opponent Scheduler

**Status: PASS** — 20 / 20 completion gates true. Machine-readable record:
`reports/phase_9_data/agent_02_population.json` (the `phase9_population_v1` /
`phase9_rollout_schedule_v1` document, digests, policy tokens and sample
scheduled records), `agent_02_schedule_audit.json` (the exhaustive audits),
`agent_02_canonical_schedule_summary.csv` (1,500 rows: iteration × bucket ×
opponent × learner colour, summing to exactly 122,880 games), and
`agent_02_acceptance.json` (verification, storage handoff, gates, Agent 3
handoff). Acceptance command:
`python scripts/run_phase9_agent02.py --run-pytest` (stages `verify`,
`storage`, `schedule`, `setups`, `artifacts`; each independently resumable).

This agent decided **which logical games should exist** and **where their
bytes will live**. It collected no self-play, built no rollout shard, ran no
optimizer step, simulated no engine ply, and never opened the final-test bank
with a model. Every learning-design constant came from Agent 1's frozen
contract; Agent 2 restated none of them.

### 2.1 Prerequisites verified

Agent 1 re-verified from live source, not trusted from its artifacts: status
`PASS` at 18/18 gates, contract digest
`ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34` equal to
`phase9_contract.contract_digest()`, both bank digests
(`3d28d544…` validation, `f38e4055…` test) equal to the accepted values and
to the digests recorded inside the bank artifacts themselves, the eight
canonical seeds and the nine contract identities equal to live source, and
`verify_phase9_upstream(include_library_digest=True)` clean.

Phase 8 corpus resolved **exclusively** through
`synthetic_corpus.default_corpus_root()` → `pointer_file` →
`/Users/brandonwashington/Dev/Github/stratego/gpt_agent/data/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1`,
with all three accepted digests unchanged (content `c95c3545…`, metadata
`1db0f02f…`, commit-index `32e8e18d…`). No absolute corpus path appears in
`phase9_schedule` or `phase9_storage`; the harness pins the expected path only
to check the resolver. Corpus identity remains version + digests, so a pure
relocation stays compatible and a mismatch would have been `BLOCKED`.

### 2.2 What was implemented

`stratego/training/phase9_schedule.py` — the logical layer implementing
`phase9_population_v1` and `phase9_rollout_schedule_v1`: `ScheduledGame` (the
full per-game record), the enumeration API, the pure game-ID rebuilder, the
`ActiveHistoryManifest` interface, resume subtraction, schedule documents and
digests, and the six audits. It contains **no filesystem call, no environment
lookup, no clock and no path** — an AST-level test enforces that, and it is
the structural proof that worker count, partitioning, arrival order, resume
boundary and storage location cannot reach a logical identity.

`stratego/training/phase9_storage.py` — rollout-root resolution and volume
diagnostics, deliberately a *separate* module that `phase9_schedule` never
imports. Resolution follows the frozen relocation semantics
(`rollout_store_schema()["relocation"]`, "the accepted Phase 8 relocation
precedent") verbatim: `STRATEGO_PHASE9_ROLLOUT_ROOT` → the durable pointer
file `data/phase9_rollout_root.txt` → the repository default
`data/phase9/rollouts`. No new fallback semantics were invented.

Agent-2 naming decisions, within frozen semantics and carrying no learning
meaning: `B{iteration:03d}` for the iteration's behavior snapshot (mirroring
the frozen `H{iteration:03d}` archive spelling), the stored policy tokens
`phase9_behavior_v1|ns=<namespace>|B0nn` and
`phase9_archive_v1|ns=<namespace>|H0nn`, and the namespace-free
`phase9_anchor_v1|H000` — H000 is the one accepted Phase 8 checkpoint,
bit-identical in every run, whereas pilot `H005` and canonical `H005` are
different weights and must never share a token. Rule and stress sides use
Agent 1's frozen Phase 4 `id@version` form.

### 2.3 Exhaustive enumeration

All six pilot runs and the full 60-iteration canonical run were enumerated —
**172,032 logical games**, every one of them, not a sample.

```text
canonical   60 x 2,048 = 122,880    digest bc253e8be2c63db1…
pilot_p9a    8 x 1,024 =   8,192    digest c7bf66ff84b7ed82…
pilot_p9b    8 x 1,024 =   8,192    digest dd74fb86cb631088…
pilot_p9c    8 x 1,024 =   8,192    digest 1b5c4e742ab1e7be…
pilot_p9d    8 x 1,024 =   8,192    digest b9a42c9a34222dfc…
pilot_p9e    8 x 1,024 =   8,192    digest 1e61f341c6568388…
pilot_p9f    8 x 1,024 =   8,192    digest 27a8b2566514fdd4…
population document digest          6756790b15ee6619…
```

Every canonical iteration is exactly `current 1,024 / historical 512 /
rule 307 / stress 205`, every pilot iteration exactly
`512 / 256 / 154 / 102`; rule subdivisions exactly `154/107/46` and
`77/54/23`. Zero bucket-count, rule-subdivision or stress-allocation
mismatches across all 108 audited iterations.

**Stress allocation.** 205 does not divide by six, so one policy takes a
one-game remainder each canonical iteration; the `(ordinal + iteration) % 6`
rotation moves it, and over 60 iterations every policy receives exactly
**2,050** games — long-run spread 0. A pilot's 102 divides exactly, so each
policy gets 17 every iteration, spread 0.

**Colour balance.** Learner red iff `(ordinal + iteration) % 2 == 0`.
Even-sized ranges split exactly (historical 256/256; canonical strategic
77/77); odd ranges carry a one-game remainder that alternates with iteration
parity (canonical stress 103/102 on even iterations, 102/103 on odd). Over
the whole canonical run the learner plays **30,720 red and 30,720 blue** —
gap 0. Same for every pilot.

**Historical league.** Outcome-independent throughout: no schedule function
reads a result, win rate or standing, and the archive manifest arrives as an
explicit immutable input validated against the frozen window. Iteration 1's
window is exactly `(H000,)`; iteration 60's is
`H000 + H020…H055` (anchor plus the eight most recent eligible snapshots).
Canonical draws: H000 7,954, H005 4,657, H010 3,895, H015 3,031, H020 2,745,
H025 2,266, H030 1,767, H035 1,512, H040 1,184, H045 884, H050 543, H055 282
— summing to 30,720. H000 leads because it never leaves the window, which is
the frozen rule, not a preference.

**Learner control.** `current/current` is `both` with one behavior-snapshot
identity on both sides; `current/historical`, `current/rule` and
`current/stress` train the current-policy side only and carry an explicit
learner colour. Only a rule or stress *opponent* side owns a match-level
policy RNG stream (the learner side is always null); only historical games
own an archive-draw stream.

### 2.4 Seed-collision audit

Every finite-width derived scheduling stream was audited **separately** over
all 172,032 games — **903,168 derived 63-bit seeds, zero collisions**:

```text
setup_root          172,032 values   0 collisions
setup_side_red      172,032 values   0 collisions
setup_side_blue     172,032 values   0 collisions
policy_red          172,032 values   0 collisions
policy_blue         172,032 values   0 collisions
historical_opponent  43,008 values   0 collisions
```

Also zero same-game red/blue setup-side collisions (which would have made
both players draw the identical board) and zero cross-stream shared values.
Cross-stream coincidence is reported but is not a violation: the domains are
separated and consumed by different code paths.

### 2.5 Identity, order independence and resume

Zero duplicate game ids within any iteration, within any run, or across all
seven namespaces: 172,032 ids, 172,032 distinct, **0 cross-namespace
collisions**. Worker/order independence was proved rather than assumed —
each audited iteration was rebuilt under 12 partitionings (worker counts
1/3/8/13 × round-robin, contiguous blocks, reversed blocks), every record
reconstructed from its identifier alone, **0 mismatches**. Resume is exact
set subtraction at commit depths 0 %, 37 %, 50 %, 99.9 % and 100 %: pending ∪
committed = scheduled, disjoint, every pending id rebuilt identically, and a
committed id from a neighbouring iteration or another namespace **raises**
rather than being silently ignored — the behaviour that stops a resume from
sealing an incomplete rollout.

### 2.6 Setups

Resolved for real across the **entire** schedule — 172,032 games, 344,064
setup sides — through the frozen `training_setup_source('neutral_v1')` with
`root_seed = setup_root_seed(game_id)`, `environment_id=0`, `generation=0`.
Zero split violations (every side `train`, purpose `training`), all 16
families present with counts 21,259–21,919 against an expectation of 21,504
(±1.5 %), zero games where both sides drew the same board, and — checked
board-string against board-string — **zero of the 1,233 held-out setups in
the frozen Phase 9 validation and test banks appear anywhere in the train
rollout schedule**. That bank read was a structural audit only: no model, no
inference, no game.

### 2.7 Storage handoff (diagnostic, never identity)

Determined from the live machine, not a remembered mount path:

```text
volume            Brandon_Washington   /Volumes/Brandon_Washington
device            /dev/disk5s2         USB, external, not removable
filesystem        Case-sensitive APFS  journaled, not read-only, unencrypted
capacity          931.09 GiB free of 931.3 GiB
sequential write  87.9 MiB/s (fsync'd, 32 MiB probe, byte-identical round trip)
proposed root     /Volumes/Brandon_Washington/stratego_phase9/rollouts
```

Projected requirement for all 172,032 scheduled games: 8.08 GiB — the
measured Phase 8 rate of 12,606 bytes/game (352,975,450 bytes over 28,000
committed games) scaled by a deliberately pessimistic 4× for longer neural
games and the sidecar. Measured free space exceeds that by **115.2×**,
against a required headroom of 10×. The drive is mounted, writable (proved by
an actual write/read/unlink round trip), and not read-only.

The redirect was recorded in `data/phase9_rollout_root.txt`, so
`phase9_storage.default_rollout_root()` now resolves to the external root for
Agent 3 without anyone needing to export anything. The root was created
**empty**; no rollout corpus exists. The repository-default probe directory
was removed. The accepted Phase 8 corpus was **not** relocated: it is
accepted where it is, and the external drive's availability is not a reason
to move it.

**The path is an operational diagnostic and never an identity.** Phase 9
rollout identity is rollout version + logical game ids + payload/metadata
digests + commit identities, so a rollout copied byte-for-byte to another
volume is the same rollout. This is enforced structurally (`phase9_schedule`
imports `phase9_storage` nowhere and performs no I/O at all) and checked
directly: the canonical iteration-1 digest is
`9f80eda2d9a1d2c9…` under the external configuration, under the repository
default, and under an environment override naming a volume that does not
exist.

### 2.8 Tests and completion gates

New regression files: `tests/training/test_phase9_schedule.py` (96),
`tests/training/test_phase9_storage.py` (22),
`tests/training/test_phase9_agent02_artifacts.py` (36, artifact-gated). Every
required control has a negative counterpart: out-of-range ordinals, iterations
outside the frozen budget, unknown namespaces and buckets, malformed game ids,
foreign committed ids on resume, archive manifests that disagree with the
frozen window, manifests carrying digests for identities outside it, an
unwritable storage location, and a volume without the required headroom. The
population, run-schedule and iteration digests are pinned, so any edit to the
schedule arithmetic or record shape fails the suite and invalidates the
artifacts. Suite: 3,928 passed / 3 skipped before; **4,082 passed / 3
skipped** after (266.15 s), recorded in the acceptance artifact.

All 20 completion gates true: `agent1_pass`, `contract_digests_match`,
`corpus_resolver_verified`, `corpus_digests_match`, `pilot_schedules_exact`,
`canonical_60_iteration_schedule_exact`, `canonical_total_games_122880`,
`duplicate_game_ids_zero`, `seed_collision_violations_zero`,
`bucket_count_mismatches_zero`, `rule_subdivision_mismatches_zero`,
`stress_allocation_mismatches_zero`, `color_balance_violations_zero`,
`train_setup_split_violations_zero`, `worker_order_dependence_zero`,
`resume_identity_mismatches_zero`, `setup_family_coverage_complete`,
`storage_root_resolved`, `no_neural_training`, `full_suite_green`.

### 2.9 Handoff to Agent 3

`agent_02_acceptance.json → handoff_to_agent_3` names the schedule
enumeration API, the pure game-ID parser/rebuilder
(`rebuild_scheduled_game`), the `ActiveHistoryManifest` interface, the
`learner_control` field, the exact policy/checkpoint identities, the setup
identity derivation, resume subtraction, and every digest above. Separately,
`storage_handoff` carries the logical schedule identity + digests **and**,
distinctly, the resolved rollout root, the measured free space, the
resolution source, and the statement that the path is diagnostic rather than
identity.

**Carry-forward notes (recorded in the artifact, not mine to resolve).**
Agent 1's `active_historical_window()` applies the frozen 5-iteration archive
cadence in *every* run namespace, so each 8-iteration pilot schedules `H005`
opponents from iteration 6 onward (359–383 games per pilot). The common contract
states the cadence under the canonical run only; the scheduler follows Agent
1's namespace-independent function rather than inventing a pilot-specific
rule. **Agents 5/6 must therefore archive an immutable pilot snapshot after
pilot iteration 5**, or those scheduled games have no opponent weights to
load. Second: the recommended rollout root sits on a USB volume — if it is
unmounted at collection time the resolver still returns it through the
pointer file and the write fails loudly rather than silently landing on the
boot disk; logical scheduling is unaffected either way.

One carry-forward Agent 3 must act on: `GameRecord.collection_checkpoint_id`
identifies the **iteration's current behavior snapshot**, but a historical
neural opponent's decisions were produced by a *different* checkpoint whose
identity travels in the `phase9_rollout_store_v1` sidecar
(`opponent_checkpoint_sha256`, surfaced as
`ScheduledGame.opponent_checkpoint_digest`). Agent 3 must verify each neural
decision against the acting side's own checkpoint identity, never against the
game-level behavior digest.

Agent 3 must collect exactly these games — no more, no fewer.

## 3. Agent 3 — Self-Play Collector and Crash-Safe Rollout Store

**Status: PASS** — 24 / 24 completion gates true. Machine-readable record:
`reports/phase_9_data/agent_03_rollout_store.json` (the store schema, the
per-iteration seal results, the replay/provenance audit),
`agent_03_collection_soak.json` (the 8,192-game soak, per-iteration and
total, with the measured storage density and its projection),
`agent_03_behavior_reproduction.json` (the reproduction audit and its
negative control), and `agent_03_acceptance.json` (verification, storage,
observer safety, crash/resume, gates, Agent 4 handoff). Acceptance command:
`python scripts/run_phase9_agent03.py --run-pytest`, followed by
`python scripts/run_phase9_agent03.py --record-final-suite`.

This is the first agent allowed to run neural self-play. It ran **no
optimizer step, computed no loss, took no gradient, and trained on nothing** —
a claim checked structurally rather than asserted (§3.9). The Phase 9
final-test bank was never opened.

### 3.1 Prerequisites verified

Agents 1 and 2 re-verified from live source rather than trusted from their
artifacts: both `PASS` (18/18 and 20/20), `phase9_contract.contract_digest()`
= `ad3dba3c4b7b461e90b3e2f8bc08d5fd3754662fbdf27bc60e75eab27e191b34`,
`population_digest()` = `6756790b15ee6619…`, and all seven run-schedule
digests equal to Agent 2's accepted values. Exact scheduled counts re-derived
live: canonical 2,048/iteration (current 1,024, historical 512, rule 307,
stress 205), each pilot 1,024/iteration. Phase 8 checkpoint SHA-256
`f7e9c40d…ec7ca` re-hashed from the file, loaded through the normal path,
863,959 parameters, C1 config digest `31ca84ab…fe07d` re-derived.

The accepted Phase 8 corpus resolves **only** through
`synthetic_corpus.default_corpus_root()` and all three accepted digests
verify unchanged (content `c95c3545…`, metadata `1db0f02f…`, commit-index
`32e8e18d…`). An additional check greps the five Phase 9 modules for a
hard-coded absolute corpus path or `/Volumes/` literal and requires zero
hits — collection and rollout-store code names no location, ever.

### 3.2 Storage: the external volume proved before a byte is written

`phase9_storage.default_rollout_root()` →
`/Volumes/Brandon_Washington/stratego_phase9/rollouts` through the recorded
pointer file. Because a plain directory of that name on the boot disk would
satisfy every path check and silently absorb production shards, the volume is
proved four independent ways before any shard exists:

```text
diskutil            Internal = false, mounted, USB, case-sensitive APFS,
                    volume not read-only
os.path.ismount     true — it is its own mount point
st_dev              differs from the boot filesystem's device
write probe         a byte written, read back and removed
```

Free space 931.09 GiB against a projected requirement, at 10× headroom.
Any failure is `BLOCKED`; no substitute path on internal storage is ever
created. Storage remains diagnostic: the sealed rollout digest is the SHA-256
over the sorted committed `(game_id, payload_sha256, metadata_sha256)`
triples, and a test copies a sealed iteration to a second root and requires
the identical digest.

### 3.3 What was implemented

`stratego/training/phase9_behavior.py` — immutable behavior snapshots and the
neural decision path. A `BehaviorSnapshot` binds a *logical* identity
(`B001`, `H000`) to a *real* file SHA-256 and refuses to load when the two
disagree or when the checkpoint does not exist; `assert_frozen()` re-checks
both `requires_grad` and a live state-dict digest at collection start and
again at seal. The frozen temperature-1 legal softmax, the ascending-absolute
storage order, the cumulative-walk selection rule, and the batched
reproduction audit all live here.

`stratego/training/phase9_rollout_store.py` — `phase9_rollout_store_v1`. The
accepted Phase 8 `warmstart_corpus_commit_v1` protocol reused in shape
(payload → metadata → commit, each flushed in that order; recovery truncates
every file back to what the last commit claims), re-keyed to
`(namespace, iteration)` with the frozen Phase 9 sidecar fields, a persisted
state machine, random-access reader, and the seal rule.

`stratego/training/phase9_collector.py` — the collector: per-side snapshot
resolution, the lockstep multi-game runner that batches neural decisions, the
frozen Phase 4 rule/stress decision path, the observer-safety probe, and the
iteration driver (reconcile → subtract → regenerate → seal).

### 3.4 Two decisions the specification left open

**Where the sampling walk reads its probabilities.** The contract freezes
"walk legal actions in ascending action-id order accumulating behavior
probabilities". `trajectory_v1` requires `legal_action_ids` ascending in the
engine's **absolute** frame while the network scores **perspective-normalized**
squares, so for blue the two orders are different permutations of the same
set. The stored order is the only one a later verifier can read, so the walk
is over the stored ascending-absolute entries; `behavior_distribution()` is
the single place the two frames are reconciled.

Relatedly, the walk consumes the **float32-rounded** probabilities — the bytes
that will actually be stored — not the float64 values before rounding.
Walking the pre-rounding values would leave a ulp-wide window per decision in
which a verifier reading the sealed record selects a different action; across
~1.5M decisions that is a real expected mismatch, not a theoretical one. This
choice makes the record self-verifying: the audit reproduces the realized
action from the bytes alone, and the checkpoint recomputation separately
proves the distribution those bytes claim. Measured result: **zero** action
mismatches over 120,104 audited decisions.

**A fixed inference batch shape.** The collector batches decisions from many
in-flight games into one forward pass, and a resumed run does not group them
the same way. Measured on this machine with the accepted C1 checkpoint:

```text
variable batch shape   policy logits bitwise stable, value logits differ by
                       ~9e-8 (WDL softmax ~3e-8) between batch 1 and batch 8
fixed batch shape      bitwise identical for a given row at any position,
                       with any neighbours, on both CPU and MPS
```

A ~1e-8 drift sits far inside the frozen 1e-4 reproduction tolerance but
**outside float32 storage rounding** — it changes stored bytes, changes
payload digests, and would break "a resumed run converges to the same sealed
rollout digest". Every forward pass is therefore padded to a fixed row count
(production 64) carried by the snapshot; the rollout state records the shape
and the device, and a resume that would change either is refused with an
explicit error. This is a recorded collector parameter, not a contract
change, and it is a *deviation* only in the sense that the specification did
not anticipate needing it.

### 3.5 The soak schedule, and why it is this one

Iteration 1 of all seven run namespaces — canonical (2,048) plus the six
pilots (1,024 each) = **exactly 8,192 scheduled games**, no additions and no
replacements.

That schedule was chosen for one reason: `active_historical_window(1)` is
`("H000",)` in every namespace, and `H000` is the accepted Phase 8 checkpoint
— a real, immutable file with a real SHA-256. An iteration past the frozen
5-iteration archive cadence would schedule `H005` opponents that do not yet
exist, and the only way to collect those would be to invent a checkpoint
digest. The contract also defines every run's `B001` as a fresh start from
the accepted Phase 8 checkpoint, so all seven behavior snapshots are that one
real file under seven logical identities. Nothing was fabricated.

```text
current      4,096      historical   2,048
rule         1,231      stress         817
                        total        8,192

rule tiers   strategic 616   tactical 431   basic 184
stress       six policies, 136–137 games each
learner      both 4,096   red 2,047   blue 2,049
```

### 3.6 Soak measurements

Single clean pass, MPS, batch shape 64, 96 games in flight, one process.

```text
games committed              8,192   (7 sealed rollouts)
decisions                1,526,576   of which neural 1,317,546
learner decisions        1,133,311
mean game length             186.3 plies
terminal results         red 4,048 / blue 4,009 / draw 135

wall time                    887.6 s
games/s                       9.23
positions/s                1,719.9
CPU                        597.5 s = 0.673 cores busy (14 logical)
peak RSS                     584.8 MiB
peak MPS driver alloc         72.7 MiB
```

Storage density, measured rather than inherited:

```text
total on disk            221,976,464 B   (shards 183.9 MB, metadata 33.4 MB,
                                          journal 4.6 MB)
bytes/game                  27,096.7
bytes/decision                 145.4
compression ratio             0.6875  (267.5 MB uncompressed → 183.9 MB)
storage/hour                858.6 MiB
```

Projected onto the real Phase 9 workloads: canonical run (122,880 games)
**3.10 GiB**, all six pilots (49,152 games) **1.24 GiB**, Phase 9 total
(172,032 games) **4.34 GiB**. Agent 2's 8.08 GiB figure was a planning
estimate scaled from the Phase 8 rule-vs-rule corpus with a deliberately
pessimistic 4× factor; the measured neural-rollout density is **0.54×** that
estimate. Both are reported side by side in the artifact, and the measured
figure supersedes the estimate for capacity decisions.

The seven sealed rollout digests:

```text
canonical   df2e6e4485bbc6ed86dc6fc78c56a6613c189ca54d97e405fb0a60613750a17d
pilot_p9a   f0d6cae8d6c7b017664bcb4e2dc2b388d546c0272edd8f6074cebe5112fc1504
pilot_p9b   abf5d40af9e77514573f533030034ba7d6c189ad0027aeae7868e05c397a430e
pilot_p9c   6a8419fb679f3f9361b477d76205b03a46ebdfe06955f34c7ac7c533654d942b
pilot_p9d   5dd9501fcda6ec6660cb8e378ef3a7e534c0ad9326c0132ace1ceee38293e459
pilot_p9e   470db9cc1735b3f0946551dc069d14bcb5827eee2d0a4f76a9f4ccbe9e190703
pilot_p9f   5170bc8aa530ade34cc3a0bacfb3db21f120fd43d580fd5cc190793825fd31e7
```

The soak was in fact run twice from an empty directory, the second time after
the state-document fix in §3.8. All seven sealed digests, the decision count
(1,526,576) and every payload byte reproduced **exactly** — an unplanned but
useful independent check that collection is deterministic end to end, not
merely resumable.

The on-disk *total* differed between the two runs by 11 bytes, entirely in the
commit journal: a commit record carries `committed_unix`, whose decimal
representation varies in length. That is precisely the distinction the store
is built around — wall-clock bookkeeping is outside rollout identity, the
digest is over `(game_id, payload_sha256, metadata_sha256)` only, and the two
runs are therefore the same rollout despite not being the same directory
byte-for-byte.

### 3.7 Behavior reproduction audit

Every audited decision is replayed from its own sealed payload through the
frozen engine — observation, legal set and action frame are re-derived, not
trusted — and then re-scored under **the acting side's own checkpoint**.

```text
learner-controlled decisions      100,029   mismatches 0
historical-opponent decisions      20,075   mismatches 0
legal-set mismatches                    0
max |p_stored − p_recomputed|         0.0
max |WDL_stored − WDL_recomputed|     0.0
games replayed                      1,237
distinct observations             224,366
```

The maxima are exactly zero rather than merely within the 1e-4 tolerance,
because the audit re-scores at the same fixed batch shape on the same device
and the stored value *is* the float32 the walk consumed. Audited per
decision: acting player, observation digest, legal set, action frame,
behavior distribution, sampled action legality, WDL output, and behavior
snapshot identity.

The historical rows matter as much as the learner rows: they are verified
against `H000`'s own digest via the sidecar's `opponent_checkpoint_sha256`,
never against `GameRecord.collection_checkpoint_id`, which names the
iteration's current learner only.

**Negative control.** An audit that cannot fail is not evidence. A sample of
learner decisions was re-scored against the *untrained* Phase 8 checkpoint
(`01c907ee…`) with its identity substituted so the hard digest veto is
bypassed and the numerical comparison actually runs. Required and observed:
**every** decision fails, with a probability difference orders of magnitude
above tolerance.

### 3.8 The store, crash safety, and observer safety

Seal preconditions, all enforced and all separately tested with a planted
failure: exact scheduled game count, all schedule IDs match, no duplicates,
no unscheduled games, no orphan records, every payload decodes and validates,
every game replays legally, one behavior identity and one behavior checkpoint
across the whole iteration, and setup provenance reconstructs. Sealed
rollouts are immutable — a writer refuses to open one and the state machine
refuses `SEALED → COLLECTING`.

Crash injection covers all seven points (`before_payload`, `after_payload`,
`after_metadata`, `before_commit_flush`, `after_commit`, `shard_rollover`,
`between_games`); after each, recovery exposes exactly the committed games,
discards the uncommitted tail, and every survivor still decodes and
digest-checks. A live demonstration on the production store crashes a
12-game rollout after 4 commits and resumes it under a **different worker
topology** (12 → 7 in flight), requiring not just equal digests but
**byte-identical payloads** against a clean run. Observed: digest
`f99298aa5ded5691…` both times, 31,319 bytes of uncommitted work discarded,
every payload identical.

The persisted state document accumulates rather than resets. A defect found
while auditing the first soak: sealing rewrote `state.json` and dropped the
`inference_device` and `inference_batch_shape` recorded at `COLLECTING` — the
exact facts a later reader of a SEALED rollout needs, and the ones a resume is
checked against. `STATE_CARRY_FORWARD_KEYS` now carries them (and the behavior
identity, digest and collector version) through every transition, with a test
that seals an iteration and requires all of them to survive.

The observer-safety boundary is audited two independent ways, because there
are two ways privileged truth reaches a network: the array handed to the
model must equal the declared observer-safe observation and own its memory,
and permuting the true types of every opponent piece the observer may not
legally know must leave the built observation bitwise unchanged. The second
check is the one with teeth, and it is exercised by a **planted leak** — a
builder that writes hidden opponent types into a channel — which the audit is
required to detect while the frozen builder passes the identical check.

### 3.9 "No optimizer steps", checked rather than asserted

An infrastructure soak cannot prove it did not train from its own results, so
the claim is checked structurally: an AST walk over the three collection
modules requires zero *uses* of `backward`, `zero_grad`, `AdamW`, `Adam`,
`SGD`, `optim`, `optimizer`, `cross_entropy`, any `*_loss`, or `ppo`
(docstrings mentioning PPO do not trip it, since only `Name` and `Attribute`
nodes count), and a live snapshot is re-digested after playing a real game to
require zero trainable parameters and unchanged weights.

### 3.10 Tests and completion gates

Suite before any Agent 3 edit: **4,082 passed / 3 skipped** (265.9 s).
After: **4,180 passed / 3 skipped** (~285 s) with all four artifact files in
place. New tests: 23 behavior, 32 rollout store, 17 collector, 26
artifact-gated.

Because the harness necessarily runs the suite *before* it writes its
artifacts, the artifact tests skip in that pass. `--record-final-suite`
re-runs the whole suite with the artifacts present and records that result,
which is the green `full_suite_green` reports. One assertion was deliberately
removed as unsound: a test running inside the suite cannot assert that the
suite passed — it is evaluated before its own run completes, so it can only
report on a previous run and has no fixed point. The artifact tests therefore
verify every gate *except* `full_suite_green`, and that one is established by
running the suite and reading the result.

All 24 gates true: `agents1_2_pass`, `corpus_resolver_verified`,
`corpus_digests_match`, `external_volume_verified`,
`behavior_snapshot_immutable`, `one_behavior_identity_per_iteration`,
`neural_actions_legal`, `behavior_storage_matches_contract`,
`behavior_reproduction_ge_100k`, `behavior_reproduction_mismatches_zero`,
`reproduction_control_fails_on_the_wrong_checkpoint`,
`rollout_commit_protocol_pass`, `crash_resume_converges`,
`orphan_records_zero`, `duplicate_game_ids_zero`, `unscheduled_games_zero`,
`replay_illegal_actions_zero`, `setup_provenance_mismatches_zero`,
`observer_input_leaks_zero`, `collection_soak_ge_8192_games`,
`all_four_buckets_represented`, `storage_density_measured`,
`no_rl_optimizer_steps`, `full_suite_green`.

### 3.11 Handoff to Agent 4

`agent_03_acceptance.json → handoff_to_agent_4` names the sealed rollout
reader, digest-checked random-access reconstruction, the behavior quantity
(`DecisionRecord.old_probabilities`, ascending absolute order — π_b(a_t|s_t)
is its entry for `selected_action_id`), the behavior WDL outputs, the
learner-control masks and expected per-game learner decision counts, the
privileged target-only state and the Phase 6 reconstruction path that rebuilds
it, the rollout digests, the crash-safe iteration state, and the independent
reproduction API.

**The carry-forward Agent 4 must act on** is the same one Agent 2 raised, now
with a concrete API: verify every neural decision against **the acting side's
own checkpoint**. `metadata['behavior_checkpoint_sha256']` is the iteration's
current learner; `metadata['opponent_checkpoint_sha256']` is the historical
opponent's real SHA-256. `phase9_behavior.reproduce_decisions(acting_snapshot,
requests)` takes the acting snapshot explicitly for exactly this reason.
Agent 4 must reconstruct RL targets independently rather than trust collector
bookkeeping.

**Also carried forward, unchanged:** each pilot still requires a pilot-local
immutable `H005` archived after pilot iteration 5 before iterations 6–8 can
collect their scheduled historical games. Agent 3's soak deliberately does
not depend on it, but Agents 5/6 do.

**Storage note.** The soak wrote to
`<rollout_root>/agent_03_soak/` rather than into the production
`<rollout_root>/<namespace>/` tree. The games are the real scheduled
iteration-1 games and the seven sealed digests recorded in
`agent_03_collection_soak.json` are the real ones; the subtree keeps Agent 7's
canonical run starting from a namespace it created itself rather than
silently inheriting bytes. Since identity is version + digests and never a
path, adopting these bytes later is a relocation plus a digest check.

## 4. Agent 4 — RL Targets, Advantages, and Anti-Leak Audit

**Status: PASS** — 18 / 18 completion gates true. Machine-readable record:
`reports/phase_9_data/agent_04_target_audit.json` (the exhaustive audit),
`agent_04_antileak.json` (25,000 permutation trials and the five positive
controls), `agent_04_example_contract.json` (the published
`phase9_example_v1` document, its digest, and the exercised train
order/cursor), `agent_04_acceptance.json` (verification, gates, handoff).
Acceptance command: `python scripts/run_phase9_agent04.py` followed by
`--record-final-suite`. **No optimizer was constructed, no loss computed, no
gradient taken, no checkpoint selected, and the final-test bank was never
opened.**

### 4.1 Prerequisites verified

Agents 1, 2 and 3 re-read from their acceptance artifacts: all three `PASS`,
18/18, 20/20 and 24/24 gates true, zero recorded problems. The live contract
digest recomputed to `ad3dba3c…` and the live population digest to
`6756790b…`, both equal to the accepted values; all seven run-schedule digests
recomputed equal to the ones Agent 3 recorded. The Phase 8 anchor hashes to
`f7e9c40d…`.

The corpus resolver check is the mandatory one:
`synthetic_corpus.default_corpus_root()` resolves through the tracked pointer
file to the accepted root, and all three digests (`c95c3545…`, `1db0f02f…`,
`32e8e18d…`) recomputed equal to the accepted identity. Agent 4 consumes no
corpus payload — its examples come from sealed Phase 9 rollouts — so the
requirement is the resolver plus the identity. Neither `phase9_targets.py` nor
`phase9_antileak.py` contains an absolute data or rollout path.

**The audited rollout is proved to be Agent 3's sealed one before anything is
audited.** `canonical` iteration 1 of the soak subtree: state `SEALED`, 2,048
games, digest recomputed from the committed bytes to
`df2e6e4485bbc6ed…` — equal both to Agent 3's recorded soak digest and to the
digest in the iteration's own `state.json` — collected from behavior snapshot
`B001` bound to the anchor SHA-256.

### 4.2 What was implemented

Two modules, plus the harness:

```text
stratego/training/phase9_targets.py    same-player extraction, scalar value,
                                       advantages, WDL lambda targets, the
                                       per-iteration filter, the example, the
                                       batch boundary, train order + cursor,
                                       the independent example audit
stratego/training/phase9_antileak.py   paired hidden-permutation trials, the
                                       model-input boundary audit, the five
                                       positive controls
scripts/run_phase9_agent04.py          the acceptance harness
```

`phase9_example_v1` is an **Agent 4 addition, not a tenth Agent 1 identity**.
The assignment requires an example/batch contract and Agent 1 froze none; the
document names only the *shape* of the object Agent 5 receives. Every learning
constant it quotes (γ = 1.0, λ_A = 0.5, λ_V = 0.8, Q₇₅, floor 0.01, ε = 1e-8,
log floor 1e-12, minibatch 512) is read from the frozen `phase9_contract`, so
a tuned value would have to be tuned there, where the accepted contract digest
catches it. Document digest:
`a6b17a94449ab764d4b5dd054d677096adfa70c52631865499a60a7a3f44af61`.

**Two passes, and why.** τ is a per-sealed-iteration statistic, so one
decision's PPO eligibility depends on every other learner decision in the
rollout. Pass 1 derives every sequence and every advantage from the stored
decisions alone — no engine replay, no observation — which is what makes the
iteration-level filter affordable (6.7 s for 2,048 games). Pass 2 replays,
builds and audits, streamed rather than materialised (84.2 s, 3,354
decisions/s).

**The privilege boundary is a function, not a convention.**
`MODEL_INPUT_FIELDS` is `("observation",)`; `model_input_fields_only(example)`
and `build_batch(...)["model_input"]` are the only routes to the backbone, and
`phase9_antileak.audit_model_input` refuses anything else. The legal mask is a
masking input (legality is public); belief labels are targets built *after*
the public observation exists.

### 4.3 The exhaustive target audit

Every learner decision of the sealed rollout — **282,414 of them across 2,048
games, one example each** — audited with zero mismatches of any kind:

```text
learner control        both 1,024   blue 513   red 511
buckets                current 1,024  historical 512  rule 307  stress 205
learner decisions      red 141,226   blue 141,188
advantage mismatches                         0
WDL target mismatches                        0
value-target simplex failures                0
belief-target mismatches                     0
eligibility mismatches                       0
standardization mismatches                   0
same-player sequence problems                0
behavior-quantity problems                   0
example audit problems                       0
```

The recomputation is genuinely independent at every step the assignment
names. The learner designation is rebuilt from the *schedule*
(`rebuild_scheduled_game(game_id).learner_sides`) and compared against the
sidecar rather than read from it; the final-perspective outcome is recomputed
from `terminal_result`; the same-player links are re-derived by rescanning the
payload for that colour's plies and asserting nothing of that player's was
skipped between two entries; the deltas, advantages and W/D/L targets come
from arithmetic written in the harness from the assignment's formulas, with
λ_A and λ_V as literals so a tuned contract constant would surface as a
disagreement instead of propagating into both sides; the threshold is
recomputed with `numpy.quantile(..., 0.75, method="linear")`; the belief
labels are rebuilt square by square from the privileged piece records instead
of through `dense_belief_target`; and the stored learner quantity is checked
against a count taken from the payload.

The measured filter over the real rollout:

```text
tau                    0.12881821608195476   (Q75 of |A|, above the 0.01 floor)
eligible               70,604 / 282,414      retention 0.250002
mean (eligible)        0.009545094610122616
std (eligible, ddof=0) 0.300811239439604
advantage range        [-1.9704856, +1.9100954]   mean 0.002052
belief supervision     6,337,120 squares, 22.44 per decision
```

The harness's independent threshold, eligible count, mean and std agree with
the production statistics to the last recorded digit.

**Zero-variance and empty-subset are frozen, not incidental.** With an empty
PPO subset the moments are 0/0 by convention and every standardized advantage
is 0.0; with zero variance the numerator is 0 for every eligible decision, so
the quotient is again 0 and the PPO term contributes no gradient. Both cases
are recorded as explicit flags on `IterationTargetStatistics` and covered by
tests, so a later agent cannot rediscover them as a NaN.

### 4.4 Anti-leak trials

**25,000 valid hidden-identity permutation trials** over 139 real games,
24,617 of which actually reassigned an identity (a position with fewer than
two unresolved opponent pieces has nothing to permute and is counted, not
claimed), mean 22.4 hidden pieces per trial, 461 trials/s.

Each trial rebuilds the **whole example** from the counterfactual privileged
state through the production builder, rather than comparing a hand-picked
subset. That matters because Phase 9 adds advantages, standardized
advantages, PPO eligibility, W/D/L targets and behavior probabilities to the
object a trainer consumes, and each is a new place privileged truth could
enter. Result: **0 invariant mismatches** across observation bytes, legal
actions, model action mapping, legal mask, learner designation, every
public/behavior-derived PPO input, and the belief mask; **0 label-control
failures** — the privileged labels moved exactly when the hidden assignment
moved, and never when it did not.

All five required positive controls fire, on a real sealed decision:

```text
privileged identity planted in the observation   caught (observation rebuild)
privileged metadata on the model input           caught (boundary audit)
wrong action frame                               caught (frame inversion)
wrong value perspective                          caught (target recomputation)
wrong learner-control side                       caught (learner designation)
```

A control that would plant nothing — no unresolved opponent piece, a centrally
symmetric action that maps to itself under the opposite perspective, a drawn
terminal decision whose reversed target coincides with its own — **raises
rather than reporting a pass**. A vacuous control counted as fired is the
precise way an anti-leak suite becomes decorative, so the harness moves to a
decision that can host all five instead.

### 4.5 Behavior consistency, re-checked independently

**100,300 learner decisions** re-checked against the exact frozen `B001`
snapshot over 563 games, on MPS at batch shape 64, 2,580 decisions/s:

```text
learner mismatches                     0
max |p_stored - p_recomputed|          2.98e-08   (tolerance 1e-4)
max |WDL difference|                   0.0
action redraw mismatches               0
legal-set mismatches                   0
policy-token mismatches                0
snapshot weights moved                 no
```

This is not a call into Agent 3's acceptance function. The harness replays
each game itself from its own payload, regenerates legality itself at every
ply, recomputes the temperature-1 legal softmax itself in float64 directly
from the model-frame logits, maps it back to ascending absolute order through
the frozen frame table, and redraws the realized action with its own
cumulative walk over the stored probabilities. The only shared machinery is
the padded forward pass, which is the thing being verified against.

**The negative control still has teeth.** The same 256 decisions verified
against the *untrained* Phase 8 checkpoint (`01c907ee…`) fail by
**0.975** — four orders of magnitude outside tolerance — so the passing
audit is evidence rather than tautology.

**On the per-side carry-forward.** 2,048 historical-opponent decisions were
verified against `H000` with zero mismatches. Iteration 1's `H000` and `B001`
are the same accepted Phase 8 file, so that direction proves the *resolution
path* — that a historical opponent's moves are attributed to the archive
member, through `metadata['opponent_checkpoint_sha256']` — rather than a
weight difference. The untrained control above is the direction that can fail
on weights, and does. The carry-forward stays live for Agents 5-7, where an
archived `H005` really is a different network.

### 4.6 The dataset handed to Agent 5

The train order was exercised, not described. Over the real universe of
282,414 learner decisions: 552 minibatches per epoch of 512, final partial
batch 302 and consumed; two epochs draw different reproducible orders from
domain-separated seeds (`train_order_seed(namespace, iteration, epoch)`,
4440477430039106730 and 4095132215534447638); an epoch is a permutation of
the universe; and a cursor advanced seven minibatches into epoch 0 and then
resumed rebuilds exactly the keys the interrupted batch would have held.
`Phase9MinibatchCursor` carries only logical state — epoch, minibatch index,
examples consumed — with no tensors, file offsets or worker identity, which is
what makes the resume exact from the sealed rollout alone.

### 4.7 Tests and completion gates

Suite before any Agent 4 edit: **4,180 passed / 3 skipped** (258.2 s). After:
**4,280 passed / 3 skipped** (262.4 s) with all four artifact files in place.
New tests: 59 targets, 15 anti-leak, 26 artifact-gated. As with Agent 3, the artifact
tests verify every gate *except* `full_suite_green` — a test running inside
the suite cannot soundly assert that the suite passed — and that gate is
established by `--record-final-suite`, which re-runs the suite with the
artifacts present and writes `covers_agent_04_artifact_tests`.

All 18 gates true: `agents1_3_pass`, `corpus_resolver_verified`,
`corpus_digests_match`, `same_player_sequence_audit_pass`,
`red_blue_perspective_audit_pass`, `advantages_exhaustively_match`,
`wdl_targets_exhaustively_match`, `advantage_filter_exact`,
`value_target_simplex_failures_zero`, `belief_target_mismatches_zero`,
`behavior_reproduction_ge_100k`, `behavior_reproduction_mismatches_zero`,
`hidden_permutation_trials_ge_25000`, `model_input_leak_mismatches_zero`,
`positive_controls_fire`, `learner_control_mismatches_zero`,
`no_meaningful_rl_training`, `full_suite_green`.

`no_meaningful_rl_training` is checked structurally rather than asserted: an
AST walk over both new modules for optimizer/loss symbols (names and
attributes only, so prose in a docstring does not trip it) finds none, and a
snapshot used for a re-check comes back with zero trainable parameters and an
unmoved state-dict digest.

### 4.8 Handoff to Agent 5

`agent_04_acceptance.json → handoff_to_agent_5` names the deterministic
rollout-to-example iterator (`iter_rollout_examples(reader, statistics)` —
games ascending by `game_id`, decisions ascending by ply, every read
digest-checked), the example schema, the train order and its cursor
(`minibatch_keys()` rebuilds an interrupted batch's exact keys from the sealed
rollout alone), the PPO eligibility rule and its per-example flag, the
standardized advantages with both degenerate cases pinned, the stored behavior
quantity for the ratio and the full legal distribution for the KL term, the
W/D/L and belief targets, and the model-input boundary. Agent 5 implements
optimization only.

**Carried forward unchanged:** each pilot still requires a pilot-local
immutable `H005` archived after pilot iteration 5 before iterations 6-8 can
collect their scheduled historical games.

**Deviation recorded.** The exhaustive audit runs on Agent 3's sealed soak
subtree (`<rollout_root>/agent_03_soak/canonical` iteration 1) because that is
where the only substantial sealed Phase 9 rollout currently lives. The games
are the real scheduled iteration-1 games and the sealed digest recomputed here
matches Agent 3's record, so the audited object is the real one; identity is
version + digests, never a path.

## 5. Agent 5 — PPO Trainer, Dynamic Damping, Checkpoint/Resume, and Throughput

**Status: PASS** — 26 / 26 completion gates true. Machine-readable record:
`reports/phase_9_data/agent_05_trainer_contract.json` (the trainer/loss/
checkpoint contract, the constructors and hard-veto counters Agent 6
receives), `agent_05_resume_validation.json` (the CPU and MPS resume proofs),
`agent_05_stability_soak.json` (the non-selection soak, the two-checkpoint
binding fixture, the archive rehearsal and the throughput probe),
`agent_05_training_benchmark.csv` (complete iteration wall time by phase), and
`agent_05_acceptance.json` (verification, gates, handoff). Acceptance command:
`python scripts/run_phase9_agent05.py` followed by `--record-final-suite`.
**No pilot was selected, no candidate compared with another, no validation
score computed, and the final-test bank was never opened.**

### 5.1 Prerequisites verified

Agents 1-4 re-read from their acceptance artifacts: all four `PASS`, 18/18,
20/20, 24/24 and 18/18 gates true. The live contract digest recomputed to
`ad3dba3c…` and the live `phase9_example_v1` digest to `a6b17a94…`, both equal
to the accepted values; the Phase 8 anchor hashes to `f7e9c40d…`.

`synthetic_corpus.default_corpus_root()` resolves through the tracked pointer
file to the accepted root and all three digests (`c95c3545…`, `1db0f02f…`,
`32e8e18d…`) recomputed equal to the accepted identity. Phase 9 rollout
storage resolves through `phase9_storage.default_rollout_root()`. The harness
scans all three new modules for an absolute data or rollout path and finds
none.

### 5.2 What was built

```text
stratego/training/phase9_loss.py        the objective, and nothing else
stratego/training/phase9_checkpoint.py  phase9_checkpoint_v1 + the archive
stratego/training/phase9_trainer.py     phase9_trainer_v1, the MPS optimizer path
scripts/run_phase9_agent05.py           the acceptance harness
```

The loss module imports its clip epsilon, loss weights, log floor and KL
direction from `phase9_contract`; the trainer imports `OPTIMIZER_CONSTRAINTS`
and re-checks every value at construction. A constant tuned anywhere but the
frozen contract fails before an optimizer exists.

### 5.3 The objective, exactly as frozen

```text
L = L_PPO + 0.5*L_value + 0.25*L_belief + beta*D_KL(pi_b||pi_theta) - c_H*H
```

Two populations, kept apart structurally rather than by convention: PPO sees
only `ppo_eligible=True` learner examples; value, belief, KL and entropy see
every learner example of the minibatch regardless of the advantage filter.
Rule, stress and historical-opponent decisions receive exactly zero
policy/value/belief gradient because they are never members of the train-order
universe a minibatch is drawn from — a structural zero, not a weight of zero.
`_verify_batch` re-checks it per minibatch anyway, because a structural
guarantee nobody checks is a comment.

Decisions worth recording:

- **The stored float32 behavior distribution is used as written** — never
  recomputed on device, never renormalized — for both the PPO denominator and
  the KL reference, following Agent 3's storage-is-authority rule. That this
  is right is measurable rather than argued: at the on-policy start of a real
  iteration the behavior KL is `3.0e-08` and the mean PPO ratio `1.0000`, so
  `pi_theta` reproduces the stored `pi_b` to float32 noise. A mistake anywhere
  in the absolute→model frame reconciliation would move both by orders of
  magnitude.
- **A legal action whose stored probability rounded to exactly 0.0 contributes
  exactly 0 to the KL** — the `0 log 0` limit, not a NaN.
- **An empty eligible subset yields `L_PPO = 0` branch-free**, so the graph
  stays connected and the other four terms are unaffected.
- **Entropy is recomputed differentiably.** Phase 8's `legal_policy_entropy`
  returns floats for metrics; here entropy is a term of the loss and must
  carry a gradient.
- **The value loss is a soft-target cross-entropy.** The frozen WDL lambda
  target is a blend of three real outcomes, and an argmax would discard
  exactly the blend the target was defined to carry. A one-hot target reduces
  it to the ordinary cross-entropy, which is asserted.

Every component is reported independently, per update.

### 5.4 Iteration ownership

Only a `SEALED` rollout may be optimized, and the sealed digest is
**recomputed from the committed journal** rather than read from the manifest,
so a state document claiming a rollout it does not hold cannot authorize
training. Six verifications gate every iteration: sealed state and recomputed
digest; one behavior identity matching the state record; the on-policy
binding; population/schedule/contract identity per game; learner-control
semantics per game; and Agent 4's example/advantage/train-order versions.

The **game id is the authority** on which scheduled game a payload is, not the
metadata sidecar: bucket, ordinal and iteration are parsed from the id and the
sidecar is required to agree, so a metadata block that agreed with itself
about a bucket it was never scheduled for cannot verify.

The **on-policy binding** compares the live model's state-dict digest with the
collecting snapshot's. Training on another policy's rollout is the failure PPO
cannot detect from its own numbers, so it is checked before the first
minibatch rather than inferred from the loss.

After two epochs the iteration is marked trained and the rollout bytes are
untouched — verified by re-reading the journals from disk and recomputing the
digest, not by re-hashing the reader's in-memory copy.

### 5.5 Damping

Beta is updated once after each optimizer epoch, never per minibatch, under
the frozen rule and clamp. The controller's **partial epoch is state**, and
that is a correctness point rather than bookkeeping: an epoch's mean KL is an
example-weighted average over every minibatch of that epoch, so a run
checkpointed halfway through one has to carry the accumulated half. Held in a
local variable it silently resets on resume and the resumed run damps on the
post-resume half alone — a divergence no parameter comparison at the resume
boundary can reveal, because it first appears one epoch later. The accumulator
therefore lives in `kl_controller_state` and round-trips through the
checkpoint. A test runs one epoch uninterrupted in a single call and the same
epoch split across a checkpoint in another process, and requires the closed
epoch entries to be equal.

### 5.6 Checkpoint, resume and the archive

`phase9_checkpoint_v1` carries every field `CHECKPOINT_REQUIRED_FIELDS` names
— the tuple is read from the contract, not restated, so a field added there
becomes a field this format refuses to be written without. Writes are atomic:
`.partial` → fsync → **reload and fully validate the bytes on disk** →
`os.replace` → fsync the directory. Crash hooks at all three boundaries are
exercised: a crash before commit leaves the destination untouched (an existing
checkpoint keeps its previous contents), a crash after commit leaves a valid
file. Every rejection the mission lists has its own negative control:
truncation, integrity-digest mismatch, corpus drift, rollout-digest drift,
rollout-identity drift, behavior-snapshot drift, train-config mismatch,
population-version mismatch and cursor mismatch.

One format serves three roles — resume checkpoint, behavior snapshot, archive
member — deliberately, so a behavior snapshot is not a stripped export whose
provenance must be trusted but a complete checkpoint whose identity fields can
be compared field by field.

**Archive identity is namespace-qualified.** `pilot_p9a|H005`,
`pilot_p9b|H005` and `canonical|H005` are three different objects that share a
local archive number; they live in separate namespace directories, carry their
namespace inside the payload, and a member may not be filed under another
run's namespace. `H000` keeps its namespace-free spelling because the frozen
schedule already says so. An existing member is never overwritten, even when
the bytes about to be written would be identical.

Binding weights into a playable snapshot changed no accepted module: the model
is built from the Phase 9 payload and handed to Agent 3's
`load_behavior_snapshot` through its existing `model=` parameter, so the file
is still hashed and the logical identity is still bound to real bytes — the
loader's binding check is supplied with weights, never bypassed.

### 5.7 Resume validation

Five independent processes per backend over Agent 3's accepted sealed
`pilot_p9c` iteration 1 (1,024 games, 143,819 learner decisions, digest
`6a8419fb…`, never written to — every leg binds with `mark_training=False`):
an uninterrupted `straight` run, a `split-first` leg that checkpoints at the
split and then continues as the **donor**, a `split-resume` leg in a fresh
process, and two fresh `control` runs with no checkpoint anywhere.

**CPU (24 updates, split at 10) is bit-exact.** Batch identities, learning
rates, cursor positions and global steps equal at all 24 steps; the exact next
batch after resume; every logical state field equal; and `max_abs_diff = 0.0`
across all 66 tensors for the resumed run against the donor *and* against the
independently executed uninterrupted run. The CPU control legs also differ by
`0.0`, so the backend is run-to-run deterministic and the strict comparison
means what it says.

**MPS (160 updates, split at 60) meets the reviewer-approved backend-aware
criterion.** Every logical quantity is equal at all 160 steps and the next
batch after resume is exact. The resume boundary itself — the first
post-resume update, against a donor that entered that step from bit-identical
state — differs by `1.86e-09` and meets the *original* `rtol=1e-5, atol=1e-6`
tolerances. Over the remaining 100 updates the resumed run diverges from the
donor by `7.90e-04`, against a measured no-checkpoint fresh-vs-fresh envelope
of `1.19e-03`: an envelope ratio of **0.66**, well inside the limit of 10.

The control legs are what make the criterion honest rather than convenient:
two fresh identical MPS runs with no checkpoint anywhere already differ by
`1.19e-03` at update 160, so an independent-run bit comparison would measure
backend determinism rather than checkpoint fidelity. The resumed run in fact
differs from the independently executed straight run by `1.11e-03` — *less*
than two independent runs differ from each other. The acceptance tolerances
were frozen in the harness before they were used as a gate.

### 5.8 The stability soak (not a seventh pilot)

**2,804 optimizer updates over 5 sealed RL iterations**, 5,120 games, 717,005
learner decisions, 1,434,010 examples consumed. Zero non-finite losses,
gradients or parameters; zero illegal targets, data mismatches, checkpoint
errors, behavior-identity mismatches and rollout-identity mismatches; zero
non-finite metric rows.

| it | updates | beta after | mean KL | epoch clip |
|----|---------|-----------|---------|------------|
| 1  | 562     | 0.020     | 0.0350  | 0.2913     |
| 2  | 574     | 0.080     | 0.0372  | 0.2892     |
| 3  | 566     | 0.160     | 0.0320  | 0.2698     |
| 4  | 554     | 0.200     | 0.0281  | 0.2526     |
| 5  | 548     | 0.200     | 0.0267  | 0.2446     |

The damping loop visibly closes: as beta ramps `0.005 → 0.02 → 0.08 → 0.16 →
0.20` (the frozen clamp), the mean behavior KL falls and the clip fraction
with it. Maximum **epoch mean** KL `0.0401` against the `0.08` hard limit, and
maximum epoch clip fraction `0.2913` against the `0.75` limit. Advantage-filter
retention `0.24998`, which is the frozen 0.75 quantile behaving exactly as
specified. Policy entropy `1.273 → 1.154` (sharpening, not collapsing); mean
pre-clip gradient norm `1.098`, maximum `10.38` against the clip at `1.0`.

One distinction worth stating: the maximum *single-minibatch* KL was `0.0951`,
above `0.08`. That is not a veto and is not treated as one — the frozen hard
limit is on the mean iteration/epoch KL, which peaked at `0.0401`. The
per-minibatch figure is reported because a reader should be able to see it.

The soak is structured so it cannot become a pilot result:

```text
scope             infrastructure_soak (not pilot_candidate)
rollout root      <rollout_root>/agent_05_soak/  (outside every production
                  namespace directory)
archive root      checkpoints/phase9/agent05/archive/ (not the production
                  checkpoints/phase9/archive/, which Agent 6 must find empty)
validation bank   never opened
final-test bank   never opened
validation score  never computed
weights           left in Agent 5's work directory
```

The soak's train-config digest differs from candidate P9-C's even though the
two share a learning rate and an initial beta, because the scope is part of
the identity — a soak checkpoint cannot be mistaken for a pilot run's.

**Iteration 1 adopts Agent 3's accepted sealed `pilot_p9c` rollout.** That is
not a stale rollout: it was collected from the Phase 8 anchor and the soak
starts from the Phase 8 anchor, so it is genuinely on-policy — which the
trainer's on-policy binding verifies rather than assumes. Relocation is
checked to leave the digest unchanged. Iterations 2-5 are collected fresh from
the snapshot frozen at the end of the previous iteration, so no iteration ever
trains on another policy's rollout; the five behavior checkpoint digests are
all distinct, which is asserted.

### 5.9 A real namespace-local `H005`, and two different checkpoints

The frozen archive cadence applies to pilot namespaces, so iteration 5 archived
a real immutable `pilot_p9c|H005` (`acfdb7bb…`). It was then bound as a
playable historical opponent and used to play the **24 iteration-6 games that
actually schedule it** — the games Agent 3 flagged as having no weights until
such a member exists. Nothing was persisted: the point is that the scheduled
identity resolves to real immutable weights and the games run.

**The limitation Agent 3 could not close.** Every iteration in Agent 3's soak
was iteration 1, where the current learner `B001` and the historical anchor
`H000` are the same file — so "each side verified against its own checkpoint"
and "both sides verified against the same checkpoint" produced identical
evidence, and a swapped binding would have passed. Soak iteration 2 has a
genuinely trained learner `B002` (`81ae56ad…`) against the anchor `H000`
(`f7e9c40d…`), with different checkpoint and model-state digests. Over 16
historical games and 181 decisions per side:

```text
each side against its own checkpoint    181/181 verified, max |Δp| = 0.0
learner decisions vs opponent weights     0/181 verified, max |Δp| = 0.392
opponent decisions vs learner weights     0/181 verified, max |Δp| = 0.388
digest guard alone                        0/181, rejected before any forward pass
```

The swapped cases deliberately rewrite the recorded checkpoint digest as well,
because otherwise the reproducer's digest guard rejects them before a single
forward pass runs — and a digest string comparison is not the claim being
tested. Both defenses are measured separately. A swap fails by four orders of
magnitude against the `1e-4` tolerance. The same fixture exists in the suite,
built from the accepted anchor and the canonical untrained initialization, so
the claim is re-checked on every run without needing the soak.

### 5.10 Throughput

Complete iteration wall time, split by phase (per soak iteration, 1,024 games):

```text
collection                  124.3 - 131.5 s   (0 for the adopted iteration 1)
sealing / audit               3.24 s          (measured separately, below)
target construction           3.0 - 3.2 s     (advantages + train order)
data wait                    52.9 - 57.2 s
MPS forward                  34.6 - 35.8 s
loss                          ~9 s
MPS backward                 72.7 - 76.3 s
optimizer                     5.4 - 5.7 s
checkpoint                    0.15 - 0.21 s
validation infrastructure     0 s             (Agent 5 runs no validation)
train total                 207.9 - 221.2 s
complete iteration          ~218 s (adopted) / ~340 - 355 s (collected)
```

Aggregate: **1,336.6 examples/s, 2.61 updates/s** at minibatch 512 on MPS.
Peak process RSS 1,073 MiB (a true `getrusage` peak); peak MPS driver
allocation 4,170 MiB. `collect_iteration` seals inside the call it collects
in, so the seal's own cost was measured separately by repeating exactly what
sealing verifies — reading every journal, decoding and validating all 1,024
payloads and sidecars, recomputing the digest — without writing a state
transition: **3.24 s**, 0 metadata problems, digest equal to the state record.

The topology sweep tunes only knobs Agent 1 permits and **proves the logical
minibatch identities are identical at every worker count**:

```text
workers   examples/s   mean data wait
   1          246         1,756 ms
   4          755           184 ms
   6          937            19 ms
  10          809             0 ms
```

Six workers is the knee: the data wait is essentially hidden and adding more
costs contention. Batch digests are identical across all four topologies and
across the timed/untimed passes; device synchronization for the phase split
costs nothing measurable (≤1%). Losses are *not* asserted equal across
topologies on MPS, and that is deliberate — the train order's claim is about
which examples a minibatch holds and in what order, which is what the batch
digest measures; requiring equal losses would additionally assert backend
determinism, which Phase 8 already measured to be false on this stack. On CPU
they do come out equal, and the trainer's own topology test asserts it there.
Training order was not changed for locality anywhere.

### 5.11 Completion gates

All 26 gates true: `agents1_4_pass`, `corpus_resolver_verified`,
`corpus_digests_match`, `ppo_loss_matches_contract`,
`illegal_logit_masking_pass`, `value_loss_matches_contract`,
`belief_loss_matches_contract`, `kl_direction_and_beta_controller_pass`,
`entropy_schedule_pass`, `opponent_only_gradients_zero`, `cpu_resume_pass`,
`mps_backend_aware_resume_pass`, `atomic_checkpoint_tests_pass`,
`soak_updates_ge_2000`, `soak_several_sealed_iterations`, `nonfinite_zero`,
`illegal_targets_zero`, `identity_mismatches_zero`,
`kl_hard_limit_not_exceeded`, `clip_fraction_hard_limit_not_exceeded`,
`throughput_measured`, `checkpoint_binding_fixture_pass`,
`namespace_qualified_archive_pass`, `no_pilot_selection`,
`no_final_test_access`, `full_suite_green`.

Each gate that rests on tests names the tests that measure it, and the
selections are run separately: a gate claimed true because some larger module
passed is a gate nobody measured. An empty selection (pytest exit code 5)
fails rather than silently passing.

Suite: **4,431 passed / 3 skipped** (316 s), up from 4,280 / 3 after Agent 4 —
Agent 5 adds 151 tests across `test_phase9_loss.py`,
`test_phase9_checkpoint.py`, `test_phase9_trainer.py`,
`test_phase9_checkpoint_binding.py` and `test_phase9_agent05_artifacts.py`.

### 5.12 Handoff to Agent 6

```text
trainer            Phase9Trainer.from_phase8_checkpoint(path, config, corpus_identity,
                     topology=LoaderTopology(workers=6, prefetch=2, record_cache_size=48))
candidate config   Phase9TrainConfig.for_candidate(candidate_id, device='mps',
                     total_iterations=8)   # LR/beta read from the frozen matrix
resume             Phase9Trainer.resume(path, config=..., corpus_identity=...,
                     expected_sealed_rollout_digest=..., ...)
sealed rollout     bind_sealed_rollout(root, namespace, iteration,
                     behavior_snapshot=..., expected_model_state_digest=...)
                     then Phase9Trainer.bind_iteration(rollout)
epochs             Phase9Trainer.train_iteration()   # the frozen two epochs
snapshot           Phase9Trainer.save_behavior_snapshot(path,
                     logical_identity='B00N', rl_iteration=N)
archive            write_archive_member(Phase9Trainer.archive_member_payload(
                     local_identity='H005'), root, namespace=..., local_identity='H005')
                     then bind_archive_member(member)
topology           workers=6, prefetch=2, record_cache=48 (validated; the knee)
veto counters      trainer.counters — every trainer-side veto is a raise, and
                     the counter records that it happened
```

`agent_05_trainer_contract.json → hard_veto_counters` states which pilot veto
is raised by the trainer and which comes from elsewhere (illegal neural action
and observer safety from Agent 3's collector, the validation EWR guards from
Agent 6's own validation pass).

**Agent 6 may run only the frozen six candidates.** Agent 5 selected nothing:
the soak's scope is recorded in its train config, no validation bank was
opened, no score exists, and the soak's weights are deliberately left where
Agent 6 cannot inherit them.

### 5.13 Deviations and findings recorded

**The resume experiments read Agent 3's accepted sealed rollout and never
write to it.** Every bind outside the soak passes `mark_training=False`, so
the accepted iteration's state document is untouched.

**Two bugs that only a real multi-iteration run finds**, both fixed with
regression tests that fail without the fix:

1. The epoch KL accumulator was local to `train_iteration`, so a run
   checkpointed mid-epoch and resumed would close that epoch on the
   post-resume portion alone and damp differently from an uninterrupted run.
   It is now part of `kl_controller_state` and round-trips through the
   checkpoint.
2. `bind_iteration` did not drop the previous iteration's exhausted data
   pipeline, so a second iteration's first minibatch popped from an empty
   prefetch queue. The soak's iteration 2 is what surfaced it; a
   two-iteration test now covers the transition at both worker counts.

**A crashed soak cannot be restarted and reuse the previous attempt's
rollouts.** MPS is not run-to-run deterministic, so a re-run from the same
anchor produces different weights, and the on-policy binding correctly refuses
to train on rollouts collected by the previous attempt's snapshot. The harness
therefore takes an explicit `--reset-soak`; there is no silent adoption path.

**Soak iteration counts differ slightly** (562, 574, 566, 554, 548) because
each iteration's game set yields a different number of learner decisions. The
frozen minibatch size and the two epochs are unchanged; only the number of
minibatches a rollout contains varies.

## 6. Agent 6 — Bounded RL Pilot Selection

**Status: PASS — 24 / 24 completion gates true, following the reviewing
chat's Agent 6 review resolution** (§6.11). The six frozen pilots ran under
one identical scheduled budget, one candidate was vetoed by the frozen KL
hard limit, a unique winner (**P9-C**) was selected from the frozen
iteration-8 validation score, and `phase9_train_config_v1` is frozen.

This section was first reported as **BLOCKED — CANONICAL WALL-CLOCK CONTRACT
REQUIRES REVIEW**, because the winner-specific measured projection of the
canonical 60 × 2,048 run did not fit inside the then-frozen 12-hour
operational ceiling. That measurement was correct and is preserved verbatim
in §6.7 and in the artifact's `historical_ceiling_evaluation`. The reviewing
chat accepted the pilot selection and authorized one narrow change —
`phase9_operational_amendment_v1`, raising the *operational* ceiling from
43,200 s to 54,000 s and nothing else — under which the same unaltered
projection fits with 7,243 s of headroom. No pilot was rerun, retrained or
reevaluated.

Machine-readable record: `reports/phase_9_data/agent_06_pilot_selection.json`
(prerequisites, six candidate runs, budget semantics, veto evaluation,
selection, dual-ceiling projection, complete access ledger, review
resolution, gates), `agent_06_pilot_runs.csv` (every validation checkpoint of
every candidate with all score components), and
`agent_06_frozen_train_config.json` (the accepted document and its digest,
the amended document and its digest, the amendment, the unchanged runtime
identity, the reconciliations, the Agent 7 handoff). Acceptance commands:
`python scripts/run_phase9_agent06.py --stage
verify|pilots|selection|config|projection|artifacts` followed by
`--record-final-suite` (twice, the accepted two-pass convergence).

### 6.1 Prerequisites verified

Agents 1–5 re-read from their acceptance artifacts: all five `PASS`, with
Agent 5's 26/26 gates re-checked individually. The live contract digest
recomputed to the accepted `ad3dba3c…` and the live `phase9_example_v1`
digest to `a6b17a94…`; the Phase 8 anchor hashes to the accepted `f7e9c40d…`.
`synthetic_corpus.default_corpus_root()` resolved to the accepted root and
all three digests verified **including the full payload-byte audit**.
`phase9_storage.default_rollout_root()` resolved through the tracked pointer
file to `/Volumes/Brandon_Washington/stratego_phase9/rollouts`; the harness
additionally proved the resolved root sits on a real mount point under
`/Volumes/` (not the boot volume), read-write, with a successful write probe
and 930 GiB free — and **every pilot worker re-proved this at process start**,
as the supplementary review instruction requires. Agent 1's bitwise-verified
anchor evaluation export re-hashed to its recorded `cd0b22d2…`. All six
production pilot rollout namespaces, all six `pilot_p9*/H005` archive slots
and all six work directories were empty before the first pilot; Agent 5's
soak weights, rollouts and archive were confirmed to live outside every
production slot and nothing of theirs was inherited.

### 6.2 The frozen schedule, re-derived rather than trusted

`run_schedule_digest` recomputed for all six pilot namespaces equal to Agent
2's pinned acceptance digests. The complete Agent 2 logical schedule was then
re-enumerated per candidate — deliberately not reusing Agent 5's "24
iteration-6 games" integration wording — giving the exact H005 assignment
counts the archives must answer in iterations 6–8:

```text
              it6            it7            it8            total H005
pilot_p9a     122 (134 H000) 124 (132)      137 (119)      383
pilot_p9b     121 (135)      136 (120)      123 (133)      380
pilot_p9c     117 (139)      120 (136)      143 (113)      380
pilot_p9d     129 (127)      126 (130)      121 (135)      376
pilot_p9e     117 (139)      137 (119)      115 (141)      369
pilot_p9f     124 (132)      114 (142)      121 (135)      359
```

Historical bucket totals are exactly 256 per iteration everywhere.

### 6.3 How the pilots ran

One worker process per candidate (`--pilot-worker`), strictly in order
P9-A…P9-F. **All six candidates received the identical scheduled budget of 8
iterations × 1,024 games × 2 epochs** from the identical starting checkpoint;
five executed it in full and P9-E terminated early under the mandatory hard
KL veto (see §6.4), which is the frozen contract working, not a reduced
budget. Each candidate: fresh `Phase9Trainer.from_phase8_checkpoint` (fresh
optimizer, scheduler and KL controller), that scheduled budget, Agent 5's
validated topology `workers=6 / prefetch=2
/ record cache=48` (never retuned per candidate), collection on MPS at the
frozen inference batch shape 64 with 96 games in flight and 2 observer-probe
plies per game, per-iteration on-policy binding (`B00N` state-dict digest ==
live trainer digest, enforced before any collection), full-schedule
`bind_sealed_rollout` with the digest recomputed from the committed journal,
per-iteration `TRAINING → EVALUATED → COMMITTED` transitions before any
next-iteration game, an immutable candidate-local `H005` archived to
`checkpoints/phase9/archive/<namespace>/` after iteration 5 and bound back as
a playable opponent (weights digest re-checked against the live post-
iteration-5 learner), and full frozen validation passes after iterations 4
and 8. All six candidates started from the byte-identical anchor: one
distinct starting model-state digest across the whole matrix.

**Determinism receipt.** The freshly collected production iteration-1 sealed
digests of all six namespaces — including P9-E's, which sealed before its
training veto — matched Agent 3's pinned soak digests byte for byte
(`f0d6cae8…`, `abf5d40a…`, `6a8419fb…`, `5dd9501f…`, `470db9cc…`,
`5170bc8a…`). Same anchor, same seeds, same device, same batch shape, five
days apart, exact reproduction.

**H005 verification.** Every H005-opponent action of every iteration-6–8 game
was reproduced under the exact bound archive member (SHA-256 recorded in the
active-history manifest and in every game's metadata), plus a 4-game H000
control per iteration: for the winner, 117/120/143 games — exactly the
re-enumerated schedule — and 33,361 opponent decisions at max |Δp| **0.0**;
every other completed candidate identically exact. Archive digests:
`pilot_p9a|H005 = addc3828…`, `pilot_p9b|H005 = 9119872e…`, `pilot_p9c|H005 =
55fdc806…`, `pilot_p9d|H005 = 1cb69840…`, `pilot_p9f|H005 = 2bef86a2…`.

### 6.4 Six candidates, one veto, one winner

Frozen iteration-8 validation scores (S = 0.45·E_Strategic + 0.35·E_Tactical
+ 0.20·E_Phase8-anchor, `phase9_validation_bank_v1`, greedy single-request
float32, bootstrap seed 2026081607):

```text
candidate  LR     beta0   sched/exec  it4 S (diag)  it8 S (binding)  random  basic   outcome
P9-C       3e-4   0.005   8 / 8       0.5710        0.691602         0.9863  0.7422  WINNER
P9-F       6e-4   0.020   8 / 8       0.5894        0.644922         0.9941  0.7773  eligible
P9-D       3e-4   0.020   8 / 8       0.5574        0.629199         0.9922  0.7031  eligible
P9-B       1e-4   0.020   8 / 8       0.5649        0.605469         0.9746  0.6758  eligible
P9-A       1e-4   0.005   8 / 8       0.5733        0.604688         0.9883  0.7832  eligible
P9-E       6e-4   0.005   8 / 0       —             —                —       —       VETOED
```

`sched/exec` is the scheduled iteration budget against what the candidate
actually executed. **Every candidate was scheduled the same eight
iterations**; P9-E executed none of them to completion because the mandatory
hard veto fired inside its first iteration's first epoch. It is nowhere
described as having completed eight iterations, and its early termination is
the frozen contract behaving correctly rather than an unequal budget.

**P9-E's veto** fired at iteration 1, epoch 1: mean behavior KL 0.089105 >
the frozen 0.08 hard limit, before the adaptive controller's first
post-epoch update could react — 6e-4 with the weak initial beta is
uncontrollable from a standing start. No rescue rerun; its collection had
already completed and sealed, so its iteration-1 rollout remains on disk as
evidence (and its digest matches Agent 3's pin — see §6.3), but no optimizer
step of that iteration ever committed. P9-F proved the same learning rate *is*
controllable from beta 0.020: its first epoch grazed the limit at 0.0775,
the controller drove beta to the 0.2 clamp, and epoch KLs retreated to
0.044–0.051 — the fastest iteration-4 score of the matrix (0.5894) but a
final score well short of P9-C. Iteration-4 Random/Basic guards were treated
exactly as the supplementary instruction requires: recorded as intermediate
diagnostics (all passed anyway), with eligibility decided only by the frozen
iteration-8 pass, where every completed candidate cleared Random ≥ 0.90 and
Basic ≥ 0.60 by wide margins.

**The winner is unique on the first tie-break key** (score margin +0.047 over
P9-F); the full frozen chain (score → Strategic EWR → lower run-mean behavior
KL → higher examples/s) was evaluated and recorded for all candidates.
P9-C's final EWRs: Strategic **0.6895** (anchor baseline 0.4531), Tactical
**0.6855** (0.4551), vs-Phase-8-anchor **0.7070**, Random 0.9863, Basic
0.7422. Stress (report-only, 32-pair prefix schedule, never a score
component): 0.859–0.992 across all six stress policies. Eight iterations of
population self-play produced a policy that beats its Phase 8 ancestor 70/30
and improves +0.23 EWR against both rule tiers that matter — the Phase 9
premise demonstrated at pilot scale.

Safety across all five completed candidates: 0 illegal actions, 0 inference
failures, 0 observer-probe failures (2,044 probes per iteration), 0 identity
mismatches, 0 non-finite events, 0 torch-importing game workers, 0
checkpoint errors. The veto evaluation table in the selection artifact
covers exactly the twelve frozen veto conditions per candidate, with
measured observations for each.

### 6.5 `phase9_train_config_v1`, frozen with labeled digests

`agent_06_frozen_train_config.json` freezes the complete 39-field canonical-
run document — C1 identity and config digest, Phase 8 starting checkpoint
SHA and expected model-state digest, LR 3e-4, initial KL beta 0.005, every
fixed PPO/value/belief/advantage/behavior constant, the entropy schedule,
optimizer block, minibatch 512, epochs 2, population/schedule/store/order/
example/collector versions, the historical archive rule, validation and
archive cadences, 60 canonical iterations × 2,048 games, the operational
ceiling, the loader/collector topology (6/2/48, 96 in flight, batch shape 64,
MPS, 2 probe plies), all eight Phase 9 seeds, `phase9_checkpoint_v1`,
acceptance and bank versions with digests, and the corpus identity.

Exactly as Phase 8 taught, the digest namespaces are labeled and reconciled
field by field, never conflated. The document exists in two versions — as
accepted under the 12-hour ceiling, and as amended under the reviewed
15-hour operational ceiling (§6.11) — so three digests are recorded:

```text
train_config_document_digest           9284fbc6b0962937…  accepted document
                                                          (12 h), sha256 of
                                                          its canonical JSON
train_config_document_digest_amended   22ac552da90989dd…  amended document
                                                          (15 h) — the one
                                                          Agent 7 runs
trainer_runtime_identity_digest        77af4d45dd8b64e7…  Phase9TrainConfig
                                                          .for_candidate("P9-C",
                                                          namespace="canonical",
                                                          total_iterations=60,
                                                          device="mps").digest()
                                                          — unchanged by the
                                                          amendment (measured)
```

The reconciliation records 7 bridged fields (all equal), 11 runtime-only
fields (each bound by a document sub-object, e.g. `weight_decay` ←
`optimizer.weight_decay`), and the document-only fields the narrower runtime
object does not carry. One explicitly documented wrinkle: the runtime
`scope` field reads `pilot_candidate` because Agent 5's frozen `SCOPES` has
no canonical entry; the reconciliation states this so Agent 7 constructs the
byte-identical object rather than inventing a new scope silently.

**No trained pilot checkpoint is handed forward.** The handoff block carries
only the winning candidate id, the labeled digests, the fresh Phase 8
starting SHA and expected model checksum, all seeds, the population/archive
contracts, the validated topology, and the canonical budget/cadences. Pilot
weights remain in `checkpoints/phase9/agent06/`; the pilot-local `H005`
archives are pilot artifacts the canonical namespace never reads.

### 6.6 Access instrumentation — measured, not asserted

Every validation matchup of every candidate is logged with its bank version
and digest: **10** authorized `phase9_validation_bank_v1` accesses under the
`pilot_selection` purpose (5 completed candidates × 2 frozen passes; P9-E
none — it terminated early under the mandatory KL veto, before its
iteration-4 pass was due), all
through `check_validation_bank_access(purpose="pilot_selection",
phase9_agent=6)`. The sealed `phase9_test_bank_v1` appears in zero recorded
matchups, the harness never constructs a test-bank object, and no
final-test checkpoint load exists: **final-test neural games = 0, final-test
neural checkpoint loads = 0**, by enumeration of the complete evidence
rather than assertion.

One further validation-bank access exists and is recorded in the artifact as
`access_instrumentation.harness_smoke_access`, bringing the complete ledger
to **11 validation-bank neural accesses: 10 pilot-selection passes + 1
pre-pilot harness access.** It was the smoke test that proved the evaluation
plumbing worked before the pilots stage launched: 8 games on validation setup
pairs 0–1 (4 against `random_legal`, 767 decisions; 4 neural-vs-neural),
loading **only** the Phase 8 anchor evaluation export
`checkpoints/phase9/agent01/anchor_eval.pt` (`cd0b22d2…`) — on both sides of
the neural-vs-neural pair — under the throwaway identities
`phase6_smoke_anchor_greedy` and `phase6_smoke_candidate_greedy`. Neural
inference did occur. It provided no unequal selection opportunity, and that
is provable rather than asserted:

- **zero pilot checkpoints were loaded**, so the access is symmetric across
  the matrix by construction — it cannot favour one of six candidates;
- it ran **before** the pilots stage, when no candidate checkpoint existed;
- **no score was computed** and nothing entered a journal, stage payload or
  artifact; the selection stage reads only `stage_pilot_*.json`;
- its cached games are **unreachable** by any production pass: the smoke
  policy tokens differ from every production token
  (`phase6_pilot_p9X_itN_greedy`), so match ids, schedule digests and chunk
  filenames differ, and the cache lived in the session scratchpad — a search
  of the production tree finds no smoke token anywhere;
- the sealed final-test bank was untouched.

Its only repository side effect was creating the structural bank cache
`checkpoints/phase9/agent06/validation_bank.json`, which holds no game or
inference result and is re-verified against the accepted bank digest
`3d28d544…` on every use.

*(An earlier draft of this section said "11 authorized `pilot_selection`
accesses", double-counting P9-F. There were 10 pilot-selection accesses and 1
harness access; both are now enumerated in the artifact.)*

### 6.7 Wall-clock decomposition and the canonical projection

Measured end-to-end per candidate (worker process wall clock 3,814–3,967 s ≈
64–66 min; P9-E 314 s to its veto):

```text
            collect   train    targets  ckpt  archive  H005-verify  validations
P9-A        908 s     1,581 s  23 s     1 s   0.2 s    19 s         1,326 s
P9-B        913 s     1,630 s  23 s     1 s   0.2 s    21 s         1,261 s
P9-C        924 s     1,673 s  23 s     1 s   0.2 s    21 s         1,318 s
P9-D        929 s     1,672 s  23 s     1 s   0.2 s    21 s         1,271 s
P9-F        909 s     1,628 s  23 s     1 s   0.2 s    19 s         1,228 s
```

The candidate-vs-anchor matchup runs through `play_match` with two
in-process owners fanned across 4 worker processes (~3.3 games/s vs the 0.7
games/s Phase 8 serial rate); rule and stress matchups run through
`run_neural_schedule` with 8 pure-engine workers (0 importing torch).

**The projection.** From the winner's own measured basis — 8.864 games/s
collection, 1,345.6 examples/s training, 140,738 mean / 146,685 max learner
decisions per pilot iteration, 530.6 s per core validation pass — one
canonical iteration costs 231.1 s collection + 418.4–436.0 s training + 5.9 s
targets + 0.2 s checkpoints, and the frozen run is 60 iterations + twelve
validation passes + twelve archive events:

```text
mean-decisions basis   45,697 s = 12.69 h
peak-decisions basis   46,757 s = 12.99 h
ceiling as first frozen 43,200 s = 12.00 h   -> overrun +2,497 to +3,557 s
ceiling as amended      54,000 s = 15.00 h   -> headroom 7,243 s (10 restarts)
                       (each restart adds ≈ 679 s re-execution)
```

**As originally reported.** The frozen contract simultaneously fixed 60
iterations, 2,048 games, 2 epochs, twelve validation passes and the 12-hour
operational ceiling, and forbade silently altering any of them. The measured
projection did not fit, so this agent stopped before Agent 7 and returned
**BLOCKED — CANONICAL WALL-CLOCK CONTRACT REQUIRES REVIEW**. The artifact
recorded the measured levers with no recommendation baked in: the overrun was
5.8–8.2 % of the ceiling; the twelve validation passes cost 6,367 s (14.7 %
of the ceiling); training is 63–65 % of an iteration at 2.6 updates/s on MPS.
Raising the ceiling, trimming the budget, or accepting the risk of an
incomplete-but-reported run were all reviewing-chat decisions, not Agent 6's.

**As resolved.** The reviewing chat raised the operational ceiling alone
(§6.11). The projection above is byte-identical to the one first reported —
no measurement was recomputed, re-timed or re-based — and both evaluations
are kept in the artifact, the superseded one under
`canonical_projection.historical_ceiling_evaluation`. Under the amended
54,000 s ceiling the peak-decisions projection fits with 7,243 s of headroom,
enough to absorb ten full-iteration restarts.

### 6.8 Completion gates

24 gates, all true. Two were added by the review resolution
(`validation_access_ledger_complete`, `operational_amendment_recorded`), and
`canonical_projection_within_ceiling` — the single false gate as first
reported — is now true against the amended operational ceiling:

```text
agents1_5_pass                          true
corpus_resolver_verified                true
corpus_digests_match                    true
rollout_storage_mounted_external        true
candidate_count_6                       true
unregistered_candidates_0               true
identical_starting_checkpoint_identity  true
logical_schedule_fairness_pass          true
equal_iteration_budget                  true
equal_game_budget                       true
equal_validation_schedule               true
hard_veto_logic_exact                   true
no_surviving_candidate_breaches_a_veto  true
selection_score_reproducible            true
winner_unique                           true
h005_archives_bound_and_verified        true
frozen_train_config_complete            true
frozen_train_config_digest_written      true
no_pilot_checkpoint_handed_forward      true
final_test_neural_access_zero           true
validation_access_ledger_complete       true   (added by the resolution)
operational_amendment_recorded          true   (added by the resolution)
canonical_projection_within_ceiling     true   (against the amended ceiling;
                                                FALSE against the historical
                                                12 h, preserved in the artifact)
full_suite_green                        true   (recorded by --record-final-suite)
```

The suite went from the pre-edit 4,431 passed / 3 skipped (326.5 s, commit
`8c59308`) to 4,462 passed / 3 skipped as first reported, and to the steady
state **4,493 passed / 3 skipped** (320.9 s) after the review resolution
added the amendment tests — each via the accepted two-pass
`--record-final-suite` convergence (the first pass reads 4,492/4: the
self-referential artifact test skips until the flag is written).

45 artifact-gated tests in `tests/training/test_phase9_agent06_artifacts.py`
pin the artifacts to the frozen contract (score recomputation, tie-break,
scheduled-vs-executed budget semantics, H005 coverage equal to the schedule
re-enumeration, veto-table exactness, access-ledger completeness, both
config digests' self-consistency, the single-field amendment reconciliation,
runtime-identity invariance, sealed-bank non-access, ceiling-gate/projection
agreement), and 16 unit tests in `tests/training/test_phase9_amendment.py`
pin the amendment itself — including negative controls that fail if a second
field is ever smuggled into it or if the base contract is edited. The ceiling
gate is treated as a measured outcome: the suite asserts its *consistency*
with the recorded numbers, so the suite stays green whichever way the verdict
falls.

### 6.9 Deviations and notes

- The wall-clock verdict was the one deviation as first reported; it is
  resolved by the reviewed amendment in §6.11. No frozen quantity was altered
  to make the projection fit, and the projection was never recomputed.
- The first draft of §6.6 said "11 authorized `pilot_selection` accesses",
  which double-counted P9-F. The true figure is 10 pilot-selection accesses
  plus 1 pre-pilot harness access — a labelling error in prose, not a missing
  or extra evaluation; both are now enumerated in the artifact, and a gate
  (`validation_access_ledger_complete`) plus a test now check the ledger
  against what the runs actually did.
- P9-E's `iteration1_digest_cross_check` entry reads `matches: false` only
  because a vetoed candidate has no journal-complete iteration; its sealed
  store digest was checked directly and equals Agent 3's pin (`470db9cc…`).
- The pilot harness restarts at iteration granularity (re-executing an
  uncommitted iteration against its own sealed rollout) rather than
  mid-epoch resume; Agent 5's exact-resume path remains the canonical-run
  mechanism. No restart was needed: all six workers ran straight through.
- Validation passes are file-based (`B005`/`B009` snapshots), so cadence
  position, not process lifetime, decides what runs on a restart.

### 6.10 Handoff to Agent 7

The complete handoff block is frozen inside
`agent_06_frozen_train_config.json`: winner **P9-C**, the document to run
(`config_amended`) and its digest `22ac552d…`, the accepted 12-hour document
digest `9284fbc6…` retained for the record, the unchanged runtime identity
digest `77af4d45…`, the operational amendment and its digest `ee4b0507…`,
the fresh Phase 8 starting SHA `f7e9c40d…` and expected model-state digest,
all eight Phase 9 seeds, the population and archive contracts, the validated
topology, and the canonical budget and cadences.

Agent 7 starts freshly from the Phase 8 anchor with this configuration and
nothing else, under the amended 54,000 s operational ceiling. It must
construct the runtime object exactly as recorded —
`Phase9TrainConfig.for_candidate("P9-C", namespace="canonical",
total_iterations=60, device="mps")` — whose `scope` field reads the legacy
`pilot_candidate` token because Agent 5's frozen `SCOPES` has no canonical
entry. **The canonical run is defined by `namespace="canonical"` and
`total_iterations=60`, not by the scope string**; the token is preserved
deliberately so the runtime identity digest continues to match, and this is
recorded in `handoff_to_agent_7.runtime_scope_note`. No pilot checkpoint is
handed forward and no further pilot training or model selection is
authorized.

### 6.11 Review resolution and `phase9_operational_amendment_v1`

The reviewing chat formally accepted the pilot selection — P9-C remains the
unique frozen winner, and no pilot was rerun, retrained or reevaluated — and
authorized one narrow change plus three reconciliations. All of it is
label-and-artifact work: the only stages re-executed were `config`,
`projection` and `artifacts`, which are pure functions of the stored pilot
payloads and run no games and no optimizer steps.

**The amendment.** `stratego/training/phase9_amendment.py` freezes
`phase9_operational_amendment_v1` (digest `ee4b0507…`) as a *separate*
review-authorized identity that changes exactly one operational number:

```text
canonical operational ceiling   43,200 s (12 h)  ->  54,000 s (15 h)
```

It is deliberately not an edit to `phase9_contract.py`, and the reason is
correctness rather than bookkeeping: `contract_digest()` is stamped into
every one of the 57,344 committed pilot games' `phase9_rollout_store_v1`
metadata sidecars and into every `phase9_checkpoint_v1`. Editing the frozen
contract would move that digest and every sealed rollout and checkpoint
produced by Agents 3–6 would stop verifying against the library that produced
it. So `CANONICAL_WALL_CLOCK_CEILING_HOURS` remains `12`,
`contract_digest()` remains `ad3dba3c…` (recomputed and asserted), and the
amendment sits beside the contract carrying the digest of the exact base it
amends. `verify_base_contract_untouched()` measures this on every use.

The amendment's own document enumerates what review did **not** authorize —
reruns, further training or selection, any change to the 60 iterations, the
2,048 games, the two epochs, the twelve validation passes, the archive
cadence, the P9-C hyperparameters, the population, the seeds, the selection
rule, the acceptance thresholds, or the sealed final-test bank — and its
`unchanged` manifest reads those quantities live from `phase9_contract` so a
later edit elsewhere cannot hide behind this paperwork.

**Three digests, three namespaces, never conflated.** The amendment changes
the train-config *document*, so that document has two digests and both are
recorded:

```text
train_config_document_digest           9284fbc6…  accepted document, 12 h
train_config_document_digest_amended   22ac552d…  amended document, 15 h
trainer_runtime_identity_digest        77af4d45…  UNCHANGED by the amendment
phase9_operational_amendment_v1        ee4b0507…  the amendment itself
```

The document reconciliation is computed, not described: **39 fields compared,
38 byte-identical, exactly one changed** —
`wall_clock_ceiling_hours: 12 → 15`. The harness refuses to proceed if any
second field moves.

**The runtime identity did not change, and that is measured rather than
claimed.** `Phase9TrainConfig.identity()` carries no wall-clock or ceiling
field at all, and its `contract_digest` entry reads the unmodified base
contract, so the same constructor re-run after the amendment yields zero
differing fields and the identical digest `77af4d45…`. The artifact records
that comparison (`runtime_identity_effect`), and a negative-control test
proves the comparison can detect a real change.

**Reconciliation 1 — the access ledger** is in §6.6: 10 pilot-selection
accesses (not 11) plus 1 pre-pilot harness access that loaded only the anchor
export, involved zero pilot checkpoints, computed no score, and reached no
selection input.

**Reconciliation 2 — the budget** is in §6.3 and §6.4: all six candidates
were scheduled the identical eight iterations; P9-E executed 0 of 8 and is
never described otherwise; the artifact now carries explicit
`scheduled_iterations`, `iterations_executed`, `ran_full_scheduled_budget`
and `terminated_early_by_hard_veto` fields per candidate, plus a
`budget_semantics` block stating the rule.

**Reconciliation 3 — the digests** is above.

The superseded 12-hour evaluation is preserved rather than rewritten: the
artifact keeps it under
`canonical_projection.historical_ceiling_evaluation`, verdict included, with
`status: "superseded for cause by phase9_operational_amendment_v1; the
measurement itself is unchanged and stands on the record"`.

## 7. Agent 7 — Canonical Population Self-Play Run

**Status: PASS — 27 / 27 completion gates true**, under the reviewing chat's
two operational amendments (§7.9). The one canonical Phase 9 run executed the
frozen experiment in full: 60 RL iterations, 122,880 scheduled games, two
optimizer epochs per sealed rollout, twelve validation passes and twelve
immutable archive members, started fresh from the accepted Phase 8 anchor
under P9-C's frozen hyperparameters. One checkpoint was selected by the
frozen validation score — **iteration 40, not iteration 60** — frozen to
`checkpoints/phase9/selfplay_c1_v1.pt`, and reproduced exactly through an
independent evaluation-only reload. The sealed `phase9_test_bank_v1` was
never constructed and never played.

Machine-readable record: `reports/phase_9_data/agent_07_canonical_run.json`
(identities, amendments, fresh start, execution, restart evidence, hard-stop
counters, validation history, report-only diagnostics, gates, Agent 8
handoff), `agent_07_training_curve.csv` (60 rows, one per iteration),
`agent_07_population_archive.json` (the league and its per-iteration active
windows), and `agent_07_checkpoint_manifest.json` (every checkpoint identity
and the selection record). Acceptance commands: `python
scripts/run_phase9_agent07.py --stage verify|amendment|run|freeze|artifacts`
followed by `--record-final-suite` (twice, the accepted two-pass
convergence).

### 7.1 Prerequisites verified before the first optimizer step

Agents 1–6 re-read from their acceptance artifacts: all six `PASS`. Agent 6's
status lives in `agent_06_pilot_selection.json` (24/24 gates) rather than an
`agent_06_acceptance.json`, and the harness reads it there rather than
reporting a missing file. Agent 6's winner was re-checked as unique — **P9-C**
at 0.6916015625 — and its certification that no pilot checkpoint carries
forward was required explicitly.

The live contract digest recomputed to the accepted
`ad3dba3c…` and the live `phase9_example_v1` digest to `a6b17a94…`. The Phase
8 anchor hashes to the accepted `f7e9c40d…`.
`synthetic_corpus.default_corpus_root()` resolved to the accepted root with
all three digests verified **including the full payload-byte audit**.
`phase9_storage.default_rollout_root()` resolved through the tracked pointer
file to `/Volumes/Brandon_Washington/stratego_phase9/rollouts`; the harness
proved the resolved root sits on a real mount point under `/Volumes/`, is
read-write, passes a write probe and holds more than twice the projected
canonical requirement — and **every worker process re-proved this at start**,
five times across the run. Agent 1's bitwise-verified anchor evaluation export
re-hashed to its recorded `cd0b22d2…`, and `run_schedule_digest("canonical")`
recomputed to Agent 2's pinned `bc253e8b…`.

The canonical namespace was proved clean before the run: no rollout
namespace, no archive member, no work directory and no journal existed. That
evidence is preserved in `stage_verify.json` and was deliberately *not*
overwritten when the later amendment was recorded (§7.9).

### 7.2 The frozen configuration, reconstructed rather than copied

`Phase9TrainConfig.for_candidate("P9-C", namespace="canonical",
total_iterations=60)` was built from the frozen matrix and then **required to
hash to the accepted runtime identity** `77af4d45…` before it was allowed to
construct an optimizer; a drift of any field fails there rather than training
under different physics. The amended 39-field train-config document is the one
executed, with both earlier documents preserved as historical provenance
(§7.9).

The legacy runtime field `scope="pilot_candidate"` was **measured, not
assumed**, to be inert. Reading `phase9_trainer.py` from source, the only two
branches on the value are the membership check and `if self.scope !=
SCOPE_UNIT_TEST`, which applies the *stricter* frozen-constant checks — a
production scope therefore constrains the run more than any alternative token
would and relaxes nothing. Rebuilding the runtime identity under each frozen
scope leaves all fifteen learning fields identical, and no library module
consumes `selects_a_configuration`. The audit is recorded in the artifact and
regression-tested with a control that fails when a scope-dependent learning
constant is introduced.

### 7.3 Fresh start from the Phase 8 anchor

The learner was constructed with `Phase9Trainer.from_phase8_checkpoint` from
`checkpoints/phase8/warmstart_c1_v1.pt` (`f7e9c40d…`), and its model-state
checksum was recorded **before the first optimizer update** as
`f2ec4fc24d72ca170341c2a176aec32c7bf7e75d3315bb39d365835a29d9dd8c` — equal to
both the anchor snapshot's loaded digest and Agent 6's expected starting
state. Optimizer, scheduler and KL-controller state started empty
(`global_optimizer_step = 0`, `kl_beta = 0.005`). No pilot checkpoint, pilot
optimizer state, pilot archive member or pilot rollout was read at any point;
`canonical|H0nn` is a different object from every `pilot_p9x|H0nn`.

### 7.4 Execution

```text
iterations committed        60 / 60
games scheduled            122,880
optimizer updates           79,004
training examples       40,417,342
learner decisions       20,208,671
wall clock              65,967 s (18.32 h)
worker processes                 5
```

Every iteration ran the exact frozen mixture — `current 1,024 / historical
512 / rule 307 / stress 205`, with the rule bucket subdivided `strategic 154 /
tactical 107 / basic 46` — re-verified against the sealed rollout by
`bind_sealed_rollout(require_full_schedule=True)` at every iteration. Advantage
filter retention was exactly `0.2500` in all sixty iterations, the frozen
`Q0.75` rule reproducing itself.

The KL controller behaved exactly as frozen: epoch KL above the 0.03 increase
threshold doubled beta from 0.005 through 0.02 and 0.08 to the 0.2 clamp by
iteration 3, after which epoch KL settled in the 0.026–0.042 band for the
remainder of the run. **Max epoch mean KL 0.0416 against the 0.08 hard limit;
max epoch clip fraction 0.3081 against the 0.75 hard limit.** No hard stop
fired at any point: zero illegal neural actions, zero non-finite losses,
gradients or parameters, zero behavior- or rollout-identity mismatches, zero
target reconstruction mismatches, zero checkpoint errors and zero observer
probe failures across 245,436 recorded observer probes (the collector's
session tallies, which under-count iteration 30's resumed games by the same
mechanism described in §7.6).

### 7.5 The historical league

`H000` is the Phase 8 anchor. Twelve real immutable members were created on
the frozen five-iteration cadence, each with a distinct checkpoint SHA-256 and
a distinct model-state digest:

```text
H005 f0909fad   H020 5ec66bc5   H035 a6b0ce5d   H050 fe4e72d1
H010 5b23d3b0   H025 03f2f9e2   H040 a834dc57   H055 8ec8bc86
H015 b9dded37   H030 ba7801e3   H045 dea1c7c6   H060 a684e454
```

Logical archive identity and checkpoint SHA-256 are kept as different objects
throughout: the manifest binds each real SHA to its `canonical|H0nn` identity,
and every archive member was re-bound and `assert_frozen()`-checked before use.
The active window followed the frozen rule at every iteration — anchor plus up
to the eight most recent — reaching its cap at iteration 46, after which
`H005`, then `H010`, then `H015` left active sampling while remaining stored
and immutable. No archive checkpoint was overwritten, and no outcome-prioritised
sampling exists anywhere in the path.

Historical actions were verified **against the acting archive checkpoint**, not
merely against a digest: for every iteration, every historical-bucket game's
opponent token was required to name a member of that iteration's active window
and to carry that member's checkpoint SHA, and a deterministic sample per
active identity had its opponent-side decisions numerically reproduced under
the exact bound member. All sixty verifications passed with zero failures.

### 7.6 Crash-safety and the restart exercise

Three genuine process restarts were exercised, all through
`phase9_checkpoint_v1`:

- **iteration 4, mid-epoch 1** — exited after 334 of 1,114 updates at cursor
  `epoch 0 / minibatch 334`;
- **iteration 12, mid-epoch 2** — exited after 767 of 1,096 updates at cursor
  `epoch 1 / minibatch 219`, carrying a completed epoch of controller history,
  two validation records and a three-member active archive;
- **iteration 29→30, committed boundary** — the restart required to adopt the
  second operational amendment (§7.9).

Each mid-epoch restart was verified field by field before the resumed process
was allowed another optimizer step: all seventeen logical state fields equal,
model state bitwise equal, the next planned minibatch identical, the sealed
rollout identity and behavior snapshot equal, the active archive, validation
history and best-validation record equal, and a no-grad forward probe on the
exact next minibatch. **The probe reproduced to 0.0 absolute difference on all
six loss components in both restarts** — stronger than the accepted
`phase9_backend_aware_resume_equivalence_v1` tolerance requires. The probe is a
forward pass only: it measures the backend envelope across a process boundary
and adds no training to the frozen experiment. Agent 5's donor and
no-checkpoint control legs were not re-run, because doing so would mean
executing the same optimizer steps twice on the canonical experiment.

The boundary resume was verified against the checkpoint itself, which is the
only available authority when there is no live pre-exit process: sixteen
recorded quantities compared, all equal, model state bitwise equal.

Update counts reconcile exactly across every boundary — iteration 4 committed
1,114 updates (334 + 780), iteration 12 committed 1,096 (767 + 329) — so no
optimizer step was repeated and none was skipped.

Crash-safe collection was exercised for real rather than simulated. The
boundary restart stopped a process during iteration 30's collection; on
resume, the 27 already-committed games were reconciled and kept and only the
2,021 missing games were regenerated. This is visible in the artifact: the
collector's session summary counts what that call played, so the report-only
bucket distribution is derived from the frozen mixture and the sealed-rollout
verification rather than from the collector's session tallies, with the
resumed iteration recorded explicitly.

### 7.7 Validation and checkpoint selection

Validation ran only at iterations 5, 10, … 60, only on
`phase9_validation_bank_v1` (`3d28d544…`), greedy float32 with the frozen
paired `color_swap_same_board` protocol, under an authorized
`checkpoint_selection` access every time.

```text
it   score      Strategic  Tactical  Anchor   Random  Basic
 5   0.621191   0.6133     0.6426    0.6016   0.9961  0.7266
10   0.693848   0.6641     0.7090    0.7344   0.9727  0.7715
15   0.727246   0.6895     0.7383    0.7930   0.9785  0.8301
20   0.741992   0.7051     0.7793    0.7598   0.9863  0.8379
25   0.772754   0.7383     0.7910    0.8184   0.9883  0.8672
30   0.767090   0.7598     0.7695    0.7793   0.9961  0.8320
35   0.828906   0.8262     0.8027    0.8809   0.9883  0.8633
40   0.836621   0.8320     0.8340    0.8516   0.9961  0.8516   <- selected
45   0.814551   0.8301     0.7969    0.8105   0.9922  0.8398
50   0.768945   0.7676     0.7559    0.7949   0.9902  0.8887
55   0.779102   0.7637     0.7754    0.8203   0.9941  0.8457
60   0.776074   0.7832     0.7539    0.7988   0.9805  0.8594
```

Every score recomputes from its own recorded EWRs under the frozen weights,
and the Random and Basic regression guards passed at all twelve passes.

The score rose to iteration 40 and then declined over the final four passes,
ending 0.0606 below the peak. **The frozen rule therefore decided a real
question rather than ratifying a foregone one: iteration 40 wins on strictly
highest score, uniquely, and the final iteration was not selected.** This is
the case the rule exists for.

### 7.8 The frozen Phase 9 checkpoint

```text
path                 checkpoints/phase9/selfplay_c1_v1.pt
sha256               dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea
model-state digest   f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd
selected iteration   40
source               behavior_B041.pt, copied byte-identically
```

The model-state digest differs from the Phase 8 anchor's `f2ec4fc2…`, proved
rather than assumed. The frozen file was then reloaded independently through
the evaluation-only path and re-evaluated on the same frozen validation
protocol in a directory holding no cached games, so every result was
recomputed rather than read back. All five effective win rates, the selection
score and all five results digests reproduced **exactly**, with zero policy
errors, zero illegal actions and zero inference failures.

The two evaluation exports hash differently because the evaluation container
records a `creation_timestamp`; all 66 parameter tensors are bit-for-bit
identical, and the artifact records both facts explicitly so that the SHA
inequality cannot be misread as a reproduction failure.

### 7.9 Two operational amendments, and what they did not change

The canonical run consumed 65,967 s. Agent 6's projection, measured on
1,024-game pilot iterations, was 45,697–46,757 s. The gap is an **operational**
finding: the canonical policy's self-play games lengthened materially as the
run progressed — mean game length moved from ~180 plies at iterations 11–15 to
a peak of 284.5 at iteration 46, easing to 234.6 by iteration 55 and standing at
255.4 at iteration 60 —
and both collection and training cost track that directly. This is reported as
an observed change in the runtime distribution. It is **not** evidence of
stronger play; strength claims rest on the frozen validation results above and
on Agent 8's sealed final-test evaluation.

Two review-authorized amendments raised the operational ceiling, each layered
beside what it amends rather than editing it:

```text
phase9_rl_contract_v1              12 h  43,200 s  ad3dba3c…  original, unedited
phase9_operational_amendment_v1    15 h  54,000 s  ee4b0507…  unedited
phase9_operational_amendment_v2    24 h  86,400 s  92ad4f67…  in force
```

The contract digest is stamped into every rollout sidecar and every checkpoint
the run wrote, so editing it in place would have invalidated the state the run
resumes from; `verify_chain_untouched()` measures the whole chain on every use.
The train-config document line is likewise three distinct digests — `9284fbc6…`
(12 h), `22ac552d…` (15 h), `f3b1efdb…` (24 h) — each 39 fields with 38
byte-identical and only `wall_clock_ceiling_hours` moved. The trainer runtime
identity `77af4d45…` is measured unchanged by both amendments.

Adopting v2 required a process restart, because the ceiling is bound in a
running process. It was taken at the next committed iteration boundary through
the accepted checkpoint/resume path and verified as described in §7.6. Nothing
scientific changed: the same 60 iterations, 2,048 games, two epochs, twelve
validation passes, twelve archive members, P9-C's hyperparameters, population,
schedules, seeds, selection rule and acceptance gates.

**The ceiling was a maximum, not a target.** The run ended immediately after
iteration 60's bookkeeping completed, leaving **20,433 s (5.68 h) of allowance
unspent**. No remaining time was used for additional rollouts, optimization,
validation passes, archive members or experimentation.

### 7.10 One harness fault, recorded rather than hidden

The first attempt to resume iteration 4 stopped immediately with *"the
behavior snapshot's weights differ from the live trainer weights"*. The guard
was correct for a fresh iteration start — before the first optimizer step the
learner's weights *are* the behavior snapshot — and wrong for a mid-iteration
resume, where the learner has legitimately advanced and that divergence is
exactly what PPO's ratio measures. No optimizer step was taken, repeated or
skipped; the sealed rollout and the mid-iteration checkpoint were both intact.
The harness now checks the binding `phase9_checkpoint_v1` actually records —
the resumed checkpoint must name this iteration's behavior snapshot identity,
SHA-256 and RL iteration — and omits the fresh-start comparison. The fault,
its cause, the fix and the fact that the halt was cleared are recorded in the
run journal and surfaced in the artifact as `harness_faults`, and an
end-to-end regression control at unit scale now covers the distinction.

### 7.11 Report-only diagnostics

```text
terminal results        red 61,216 / blue 60,723 / draw 914 (0.74 %)
mean game length        215.7 plies (min 177.0 at it9, max 284.5 at it46)
collection throughput   6.98 games/s mean, 4.72 min
training throughput     1,080 examples/s mean
peak RSS                1,480 MiB
stress (report-only)    draw_seeker 1.0000  information_miser 0.9922
                        chaos 0.9531  berserker 0.9219
                        scout_rush 0.8281  miner_rush 0.7969
```

Colour balance stayed even across 122,880 games. Report-only metrics rescue no
gate.

### 7.12 Handoff to Agent 8

Frozen checkpoint `checkpoints/phase9/selfplay_c1_v1.pt`
(`dfd698e5…`, model state `f1df694d…`), selected iteration 40; Phase 8 anchor
`f7e9c40d…` with model state `f2ec4fc2…`; the complete validation selection
history; the archive manifest with all twelve real SHAs; every configuration
identity including all three ceiling authorities; the training-discipline
access ledger; and all hard-stop counters at zero.

`phase9_test_bank_v1` (`f38e4055…`) is recorded as an identity only. Agent 7
never constructed a test-bank object and ran no model over it:
`test_bank_model_access = 0`. **Agent 8 owns the first final-test neural
evaluation, and performs no training.**

## 8. Agent 8 — Independent Final Acceptance and Phase 9 Freeze

### 8.1 Mission and harness

Agent 8 performed the first and only sealed final evaluation of the frozen
Phase 9 checkpoint and recommends formal acceptance. Everything ran through
`scripts/run_phase9_agent08.py` in four stages — `verify`, `discipline`,
`final`, `artifacts` — plus `--record-final-suite`. No training, no tuning,
no checkpoint replacement, no threshold change. The full pre-edit suite was
recorded first (4,582 passed / 3 skipped, commit `87fd903`).

### 8.2 Administrative freeze

Agent 7 was formally accepted subject only to freezing the exact reviewed
working tree into a stable commit before final-test access. Verified from
live git state: the tracked tree is byte-identical to HEAD `87fd903`, which
carries all four Agent 7 artifacts; the freeze was re-checked immediately
before the sealed bank opened. The only untracked entries are pre-existing
non-Phase-9 clutter (Phase 6b state dumps, an `.Rhistory`, a docs copy),
recorded in the stage evidence.

### 8.3 Identity verification from live bytes

Every identity was recomputed rather than read from handoff prose:

```text
contract                 ad3dba3c…  (12 h ceiling preserved unedited)
example contract         a6b17a94…
amendment v1 (15 h)      ee4b0507…
amendment v2 (24 h)      92ad4f67…  verify_chain_untouched() -> []
train-config documents   9284fbc6… (12 h)  22ac552d… (15 h)  f3b1efdb… (24 h, executed)
trainer runtime          77af4d45…  (measured unchanged by both amendments)
Phase 8 anchor           f7e9c40d… / model state f2ec4fc2… / 863,959 parameters
Phase 9 frozen           dfd698e5… / model state f1df694d… / 863,959 parameters
validation bank          3d28d544…  (full deterministic rebuild)
test bank                f38e4055…  (full rebuild + all-case structural audit)
Phase 7 library          7b8a6660…
corpus                   resolver + all three accepted digests
seeds                    all eight (2026081601–08)
```

Field-level reconciliation of the three train-config documents proved only
`wall_clock_ceiling_hours` differs across the chain (12 → 15 → 24); every
learning-design field is byte-identical. All 66 parameter tensors and a
train-split probe forward pass are finite.

**B041 lineage.** The frozen checkpoint's bytes are identical to
`behavior_B041.pt` — the post-iteration-40 snapshot — and its payload records
`produced_after_iteration = 40`, `snapshot_role = behavior_snapshot`, and the
collecting behavior of its source rollout as `B040`'s file SHA (`8a607394…`).
`B040` hashes differently and carries a different model state (`622ba7dc…`) —
the negative control. Archive member `H040` (created after iteration 40)
carries exactly the frozen model state — an independent second confirmation.
The selection recomputes as a strict argmax at iteration 40 (0.836621) over
the twelve cadence passes; the iteration-40 pass evaluated `behavior_B041.pt`
under the same SHA, and iteration 60 (0.776074) is correctly not selected.

### 8.4 Training-discipline evidence

```text
fresh Phase 8 start        optimizer step 0, no pilot state, anchor model state
pilot candidates           exactly six namespaces; P9-C unique winner
selection evidence         validation bank only; final-test access before Agent 8 = 0
no post-selection training frozen SHA unchanged; payload step 47,086 = journal
                           updates through iteration 40 (79,004 total); 60
                           distinct behavior snapshots B002–B061
hard stops                 all counters zero in all five sessions
```

### 8.5 Observer-safety reconciliation (resumed iteration 30)

The raw observer-probe session tally under-counts the resumed portion of
iteration 30: the completing session recorded 4,036 probes over its own
2,021 games, while the 27 games committed by the pre-restart session carried
no session summary. Reconciled from durable evidence, with nothing invented:

- **The 27 games remained valid committed games.** `w00.commit.jsonl` holds
  exactly 27 hash-anchored commits and `w01` 2,021; all 2,048 decode
  digest-clean; the sealed rollout digest recomputed from the commit journals
  equals the recorded digest and the digest training bound
  (`97b54bd5…`); the state history shows two COLLECTING entries and one
  behavior identity (`B030`) through COMMITTED.
- **The probe-count rule was measured, not assumed.** Probes fire on neural
  actors only (current and historical sides are neural; rule/stress games are
  neural only on the learner's plies, red acting first), capped at two per
  game: `probes(game) = min(2, neural plies)`. This rule reconstructs the
  recorded probe count of **all 60 iterations exactly** — including the
  handful of genuine 1–2-ply scout-to-flag games some unusual setup families
  produce, and iteration 30's own resumed portion.
- **The lost session's probes reconstruct exactly: 54** (27 games, each with
  at least two neural plies). Re-executing the probes from stored bytes gives
  54 probes, **0 unsafe**. Corrected full-run total: **245,490** (recorded
  245,436 + 54); recorded failures 0; no observer-related halt or harness
  fault anywhere in the journal.

### 8.6 The sealed final evaluation

The bank opened once, under
`check_test_bank_access("final_evaluation", phase9_agent=8)` — the first
neural access to `phase9_test_bank_v1`. Frozen protocol throughout: greedy
argmax, `single_request`, float32, `color_swap_same_board`, match root seed
20260401, bootstrap seed 2026081608 (paired-unit percentile, 10,000
replicates); the candidate evaluated under its accepted identity
(`canonical_it40`, bitwise-verified export), the anchor under `c1_warmstart`
through Agent 1's bitwise-verified export.

```text
matchup                             games   W/D/L         EWR     95% CI
candidate vs Phase 8 anchor         1,024   862/5/157     0.8442  [0.8228, 0.8647]
candidate vs strategic              1,024   828/4/192     0.8105  [0.7871, 0.8330]
candidate vs tactical               1,024   856/3/165     0.8374  [0.8149, 0.8584]
candidate vs random                 1,024   1,012/0/12    0.9883  [0.9814, 0.9941]
candidate vs basic                  1,024   873/2/149     0.8535  [0.8330, 0.8735]
anchor    vs strategic              1,024   450/10/564    0.4443  [0.4180, 0.4702]
anchor    vs tactical               1,024   475/8/541     0.4678  [0.4419, 0.4941]
```

### 8.7 Hard gates — all eight pass

```text
A  vs Phase 8 anchor   EWR 0.8442 >= 0.58     paired LB 0.8228 > 0.53      PASS
B  Strategic           EWR 0.8105 >= 0.52     improvement +0.3662 >= 0.05
                       improvement CI [0.3320, 0.3999], LB > 0             PASS
C  Tactical            EWR 0.8374 >= 0.52     improvement +0.3696 >= 0.05
                       improvement CI [0.3345, 0.4038], LB > 0             PASS
D  Random guard        overall 0.9883 >= 0.94  red 0.9902 / blue 0.9863
                       >= 0.90                paired LB 0.9814 > 0.92      PASS
E  Basic guard         EWR 0.8535 >= 0.65     paired LB 0.8330 > 0.60      PASS
F  Safety              0 illegal / 0 failures / 0 non-finite /
                       0 observer failures (11,763 probes)                 PASS
G  Policy collapse     0.0417 of 613,719 decisions > 0.999 (< 0.25)        PASS
H  Belief retention    CE ratio 0.9408 <= 0.98 [0.9396, 0.9420]
                       top-1 0.2650 > remaining-count 0.2036               PASS
```

Both stretch targets (Strategic/Tactical ≥ 0.55) are met; they remain
report-only. The paired improvements used the frozen paired-difference
bootstrap under `diff|candidate|anchor|opponent` tokens; the gate-A interval
was additionally reproduced bit-for-bit by an independent from-scratch
bootstrap implementation (`paired_bootstrap_exact = true`).

### 8.8 The replay audit behind gates F and G

Every one of the final candidate's **613,719** decisions across all 5,888 of
its final-test games (five core matchups and the six-policy stress schedule)
was replayed through the frozen model at the game-time single-request shape:
the greedy action reproduced the recorded action **613,719/613,719**, every
policy row was finite, legal-softmax max-probability exceeded 0.999 in 4.17%
of decisions (worst matchup: Basic at 6.89%), and observer-safety probes at
the frozen collection density (11,763 probes) found **0** violations.

### 8.9 Report-only diagnostics

```text
stress (64 pairs each)  information_miser 1.0000  chaos 0.9766
                        draw_seeker 0.9727        berserker 0.9531
                        scout_rush 0.9297         miner_rush 0.8555
```

Belief retention ran the accepted Phase 8 held-out benchmark unchanged
(4,000 games, 249,924 examples, 6,850,575 supervised pieces; original
remaining-count baseline, nothing refit): CE 2.0797 vs baseline 2.2107.
Phase 8-style teacher-policy imitation CE is report-only in Phase 9 and now
sits above its uniform-legal baseline (ratio 1.319) — the RL policy has
moved away from the synthetic teacher while beating every opponent tier, as
expected. Value-head diagnostics were sampled at 76,711 replayed decisions.
`agent_08_league_matrix.csv` (960 rows) aggregates the sealed canonical
rollouts by iteration and opponent: the current policy's training-time EWR
against each historical member, rule tier, and stress policy for all 60
iterations, with colour splits for the self-play bucket. Report-only metrics
rescue no gate.

### 8.10 Artifacts and final suite

```text
reports/phase_9_data/agent_08_final_acceptance.json   hard-gate table + 30 completion gates
reports/phase_9_data/agent_08_strength_results.csv    13 matchups, digests included
reports/phase_9_data/agent_08_league_matrix.csv       60 iterations x opponent identity
```

The acceptance artifact validates itself: shared logic
(`recompute_gate_booleans` / `validate_acceptance_artifact`) recomputes every
hard-gate boolean from its own observed/threshold rows both when the artifact
is written and in the artifact-gated tests, with negative controls proving a
tampered boolean, a boundary CI, or an inconsistent recommendation is caught.

Targeted Agent 8 tests: 21 harness + 18 artifact-gated. Full suite with all
Agent 8 artifacts in place: the first pass failed exactly one *new* Agent 8
artifact test that still used the stale `historical_policy` token (the same
token subtlety the probe-rule work uncovered; fixed in the test), and the
converged steady state is **4,621 passed / 3 skipped** — recorded twice, the
second time with the final PASS-state acceptance artifact on disk. The
pre-edit baseline was 4,582 passed / 3 skipped, so Agent 8 adds 39 tests and
breaks none.

### 8.11 Recommendation

Every hard gate passes, every identity and discipline gate is clean, and the
full suite is green with the artifacts in place.

```text
PHASE 9 RECOMMENDATION: PASS
```

Formal Phase 9 closure belongs to the reviewing chat.
