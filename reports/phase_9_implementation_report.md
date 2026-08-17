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
