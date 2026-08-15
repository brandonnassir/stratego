# Phase 8 Implementation Report

Synthetic warm-start training: generate games from the frozen Phase 4
rule-based population under the frozen Phase 7 setup sampler, reconstruct
policy/value/belief targets, and warm-start C1 on MPS.

Phase 8 executes against the frozen post-Phase-7 stack: rules
`stratego_project_v1`, reference engine `phase2_1_reference_1.2.0`,
observation `observation_v2_1_127ch`, engine action encoding
`source_destination_10000_v1` in absolute engine squares, model contract
`model_contract_v2` (C1 primary, 863,959 parameters; C0 fallback), backend
`KEEP_PYTHON`, trajectory `trajectory_v1` (snapshot interval 32), the
untouched Phase 4 evaluation bank `evaluation_setup_bank_v1`, and the frozen
`setup_library_v1` / `setup_sampler_v1` / `setup_source_v1` stack with the
`neutral_v1` Phase 8 profile. Phase 8 is a supervised / outcome-supervised
warm start: no self-play RL, no learned setup selection, no decision-time
search, and no engine modification is authorized anywhere below. Phase 8 is
not the official 168-hour run. Population self-play is Phase 9.

## 1. Agent 1 — Warm-Start Contract and Pre-Training Acceptance Standard

**Status: PASS** — 18 / 18 completion gates true. Machine-readable record:
`reports/phase_8_data/agent_01_warmstart_contract.json` (full contract +
verification + gates), `reports/phase_8_data/agent_01_teacher_population.json`
(roster + weights + schedule), and
`reports/phase_8_data/agent_01_acceptance_thresholds.json` (frozen gates +
pilot matrix + sealing rules). Every value below was frozen while **no
synthetic production corpus existed and no optimizer step had run**
(`data/warmstart/` absent, `optimizer_steps: 0`, `model_weights_mutated:
false`, recorded in the artifact).

### 1.1 Prerequisite verification

Verified from the repository rather than assumed:

| Check | Required | Found |
|---|---|---|
| Phase 7 formally accepted | `PASS` | `agent_06_final_acceptance.json` status `PASS`, 28 / 28 gates true, Agents 1–5 all `PASS` |
| Reference engine | `phase2_1_reference_1.2.0` | `IMPLEMENTATION_VERSION`, live |
| Rules / observation | `stratego_project_v1` / `observation_v2_1_127ch` | live constants, unchanged |
| Model contract | `model_contract_v2`, frame `perspective_normalized_squares` | live constants, unchanged |
| C1 identity | 863,959 params, digest `31ca84ab…e07d` | rebuilt live: 863,959, digest matches |
| Trajectory | `trajectory_v1`, snapshot interval 32 | live constants, unchanged |
| Phase 4 bank | digest `5fe5f987…674266`, 1,024 pairs, root seed 20260101 | regenerated live in 0.24 s, digest matches |
| Setup library | digest `7b8a6660…02777`, 8,000 entries | loaded live, digest matches |
| Sampler stack | `setup_sampler_v1`, `setup_perturbation_v1`/`seed_encoding_v1`, `setup_source_v1`, default profile `neutral_v1` (reflection 0.5, perturbation 0.5, swaps 1–6, window [2,12], 64 attempts) | live constants, unchanged |
| Phase 4 roster | exactly 10 accepted policies | live registry reproduces the expected roster exactly |

Pre-existing suite, measured at commit `144baf4` **before any Phase 8 edit**:

```text
.venv/bin/python -m pytest tests -q
3421 passed, 3 skipped, 0 failed in 202.07s
```

The three skips are the pre-existing PASS-gated capability/artifact skips
from earlier phases.

### 1.2 Frozen Phase 8 contract versions and canonical seeds

```text
warmstart_training_contract_v1    the complete learning-design contract
synthetic_warmstart_corpus_v1     corpus identity, splits, schedule
warmstart_decision_sampler_v1     deterministic per-game decision sampling
warmstart_example_v1              training-example schema
warmstart_eval_v1                 metrics, baselines, bootstrap semantics
```

Implementation: `stratego/training/warmstart_seed.py` (identity + seeds) and
`stratego/training/warmstart_contract.py` (the contract), with 80 regression
tests in `tests/training/test_warmstart_seed.py` and
`tests/training/test_warmstart_contract.py`. The serialized contract's
SHA-256 is `7b6e7b27…58de` (recorded in the artifact; later agents can pin
it).

Canonical seeds, chosen by the date-seed convention (freeze date `20260813`
+ two-digit role suffix) **before any corpus, pilot, or model result
existed**:

```text
corpus master seed          2026081301
canonical C1 init seed      2026081302
train shuffle/order seed    2026081303
pilot namespace seed        2026081304
final-run namespace seed    2026081305
validation bootstrap seed   2026081306
test bootstrap seed         2026081307
```

All Phase 8 streams derive through `derive_warmstart_seed(domain, *parts)` —
a blake2b hash under the new personalization tag `strat-ws8` (distinct from
every Phase 4/7 tag), domain-separated over `setup_root`, `policy:red`,
`policy:blue`, `decision_sampler`, `train_order`, `pilot`, `final_run` and
`bootstrap`. There is no global RNG cursor anywhere: every stream is a pure
function of identity. The canonical untrained C1 is
`build_candidate_model('C1', seed=2026081302)`; its full state-dict SHA-256
`37d1ef8d…cab2` is recorded so Agents 6–7 can prove "fresh reconstruction"
byte-for-byte.

### 1.3 Teacher population and policy-supervision weights

The live Phase 4 registry reproduces the expected roster exactly — 4 tier +
6 stress policies, all stochastic with frozen seeded selection semantics
(`decision_seed = derive_decision_seed(policy_seed, ply)`, ranking ties
broken by ascending action id):

| Policy | Version | Role | Policy weight |
|---|---|---|---|
| `strategic_rule_based` | 1.1.0 | tier_strategic | **1.0** |
| `tactical_rule_based` | 1.0.0 | tier_tactical | **1.0** |
| `basic_heuristic` | 1.0.0 | tier_basic | **0.5** |
| `random_legal` | 1.0.0 | tier_random | **0.0** |
| `stress_scout_rush` … `stress_chaos` (6) | 1.0.0 | stress | **0.0** |

Zero-weight decisions contribute no policy gradient but remain fully
eligible for value and belief supervision. No Phase 4 policy is modified.

### 1.4 Matchup schedule and corpus identity

All 100 ordered red×blue cells over the frozen roster order (`cell_index =
red_index * 10 + blue_index`), with exact scheduled counts — never natural
sampling:

```text
train         200 games / cell     20,000
validation     40 games / cell      4,000
test           40 games / cell      4,000
total                              28,000
```

Game identity is the pure function
`synthetic_game_id(split, red_token, blue_token, ordinal)`:

```text
synthetic_warmstart_corpus_v1|ms=2026081301|split=train|
    red=strategic_rule_based@1.1.0|blue=random_legal@1.0.0|g=0137
```

The runner enumerated all 28,000 identities: unique, parseable, split-exact.
Per game, four domain-separated streams derive from the id alone: the
setup-source root seed, the red and blue rule-policy match seeds, and the
decision-sampler bins. Worker count, partitioning, arrival order and resume
boundaries appear nowhere in any derivation. Corpus games run under the
frozen `TRAINING_RULES` (battleless 100 / absolute 4,000); the Phase 4
strength gates keep their frozen `EVALUATION_RULES` machinery untouched.

### 1.5 Setup sources per split

```text
train         training_setup_source('neutral_v1')          (hard-wired train)
validation    audit_setup_source('validation',
                  'Phase 8 held-out warm-start validation corpus')
test          audit_setup_source('test',
                  'Phase 8 sealed held-out warm-start test corpus')
```

Per game: `assign(root_seed=setup_root_seed(game_id), environment_id=0,
generation=0, game_id=game_id)`. Red and blue sample independently through
the sampler's frozen per-side derivation; no family receives outcome-based
weighting. The runner exercised one deterministic draw per split and the
drawn bases land in the correct library ranges (train `F14:039`/`F13:210` <
400; validation `F12:430`/`F09:441` in 400–449; test `F10:471`/`F02:496` in
450–499), with justifications recorded in provenance.

### 1.6 Decision sampler (`warmstart_decision_sampler_v1`)

Maximum 64 selected decisions per game. `T ≤ 64`: all decisions. `T > 64`:
bin `b` covers `[floor(b·T/64), floor((b+1)·T/64))`; the selected index is
`lo + (decision_bin_seed(game_id, b) mod (hi − lo))`. Bins are disjoint and
ascending, so selection is without replacement and already sorted; every
bin is provably non-empty for `T > 64`. Outcome, teacher strength, future
value and model predictions appear nowhere in selection. The runner swept 3
game ids × 15 lengths (0…4,000): 30/30 long-game selections reproduced
exactly from the published arithmetic.

### 1.7 Example schema and target semantics (`warmstart_example_v1`)

Thirteen frozen fields; **only `observation` `[127,10,10] float32` enters
the model**. `legal_mask [10000] bool` feeds the masked policy loss/adapter
only. Targets: `policy_action_model` via the frozen
`absolute_action_to_model` conversion (weight from the acting policy's
frozen contract); `value_target` WIN=0/DRAW=1/LOSS=2 from the acting
player's perspective with no Phase 8 bootstrapping; `belief_target`/`
belief_mask` are the frozen `dense_belief_target_v1` (hidden-only, model
frame, ignore index −100, 12 types). Privileged replay state may label
targets only after the public observation is constructed.

### 1.8 Baselines and evaluation contract (`warmstart_eval_v1`)

```text
policy    uniform legal: CE = ln(legal_count); expected top-1 = mean(1/legal_count)
value     one constant WDL prior fitted on train selected examples only
belief    observable unresolved-inventory marginal per hidden piece
```

Log epsilon 1e−12; top-1 ties break to the lowest index; policy metrics are
weighted by policy weight exactly as the loss is, identically for model and
baseline. Aggregation units: decision (policy, value) and hidden piece
(belief). Confidence intervals bootstrap **by game**: 10,000 replicates,
`numpy default_rng(bootstrap_seed)` index matrix, 2.5/97.5 percentiles,
model and baseline paired on the same resamples; seeds 2026081306
(validation) / 2026081307 (test).

### 1.9 Loss normalization

`L = λ_policy·L_policy + λ_value·L_value + λ_belief·L_belief`, each term
normalized over its own supervision: policy by the sum of nonzero weights
(zero-sum batches contribute 0), value by selected decisions, belief by
supervised squares (the frozen `belief_loss` normalization), so hidden-piece
count can never silently scale the belief head.

### 1.10 Pilot candidate matrix (frozen before any pilot update)

3 learning rates × 2 loss-weight profiles = **6 candidates**, at the cap:

```text
ws_pilot_lr1e-3_balanced    lr 1e-3   λ = (1.0, 1.0, 1.0)
ws_pilot_lr1e-3_policy_led  lr 1e-3   λ = (1.0, 0.5, 0.5)
ws_pilot_lr3e-4_balanced    lr 3e-4   λ = (1.0, 1.0, 1.0)
ws_pilot_lr3e-4_policy_led  lr 3e-4   λ = (1.0, 0.5, 0.5)
ws_pilot_lr1e-4_balanced    lr 1e-4   λ = (1.0, 1.0, 1.0)
ws_pilot_lr1e-4_policy_led  lr 1e-4   λ = (1.0, 0.5, 0.5)
```

Fixed for every candidate: C1, float32 on MPS, batch 256, AdamW (β 0.9/0.999,
ε 1e−8, weight decay 0.01), global-norm gradient clip 1.0, linear warmup 500
steps then constant LR, model init seed 2026081302, 5,000-update budget,
validation every 500 updates, same corpus and same selected-example
universe. Selection score = `mean(r_policy, r_value, r_belief)` on
validation (lower better) at the final pilot checkpoint; hard vetoes:
non-finite loss/gradient/parameter, target mismatch, split leak,
checkpoint/resume failure, any component ratio > 1.05. Tie-breaks: lower
score → lower policy ratio → higher examples/s. Test metrics, Phase 4
strength, architecture, teacher weights and setup sampling are outside the
matrix. Budget: ≤ 6 candidates × ≤ 5,000 updates; final run ≤ 25,000 steps.

### 1.11 Final acceptance thresholds (copied verbatim, never relaxed)

```text
Random gate      EWR ≥ 0.950 over 2,048 games (all 1,024 bank pairs);
                 both colors ≥ 0.900; paired-bootstrap 95% LB > 0.900;
                 0 illegal / 0 failures / 0 non-finite
Init gate        vs build_candidate_model('C1', seed=2026081302):
                 ≥ 512 paired cases / ≥ 1,024 games, EWR ≥ 0.700, LB > 0.550
Policy gate      test CE ≤ 0.90 × uniform-legal CE and top-1 above uniform
Value gate       test CE ≤ 0.98 × train-prior CE and better Brier
Belief gate      test CE ≤ 0.98 × remaining-count CE and better top-1
Stability gate   100% finite logits; fraction with max legal prob > 0.999
                 must stay < 0.95; entropy distribution reported
```

### 1.12 Held-out sealing (testable)

`check_test_corpus_access` / `check_phase4_bank_access` in
`warmstart_contract.py` are pure, stateless and regression-tested: before
Agent 7 the test corpus admits `structural_audit` only (model inference,
metrics, and any selection purpose raise `HeldOutAccessError`), and the
Phase 4 bank admits `non_neural_regression` only (neural playing strength
for selection raises). Agent 7 — and only Agent 7 — opens `final_evaluation`
and the final bank evaluations. Statelessness is what keeps the seal
enforceable now and still satisfiable after Agent 6 freezes the checkpoint.

### 1.13 What Agent 1 did not do

No production corpus (`data/warmstart/` does not exist), no pilot training,
no optimizer step, no model weight mutation, no engine/library/bank/roster
modification, no Agent 2 work. The held-out library draws above are
justified split-access audits (the Phase 7 Agent 6 pattern), not corpus
data.

### 1.14 Post-edit suite and completion gates

Full suite after all Agent 1 edits, artifacts present:

```text
.venv/bin/python -m pytest tests -q
3501 passed, 3 skipped, 0 failed in 199.93s
```

3,421 pre-existing tests plus the 80 new Agent 1 regressions, all green; the
three skips are the same pre-existing PASS-gated skips as before Phase 8.

18 / 18 completion gates true (recorded in
`agent_01_warmstart_contract.json`): Phase 7 acceptance verified; upstream
versions/digests match; exact 10-policy roster; 100 ordered cells;
20k/4k/4k schedule; split semantics; teacher weights; corpus identity/seeds;
decision sampler exact; target semantics exact; baselines exact; pilot
matrix ≤ 6 predeclared; selection score exact; thresholds frozen; sealing
explicit; no corpus generated; no optimizer step; suite green.

**Handoff to Agent 2** (in the artifact's `handoff_to_agent_2` block): all
five contract versions, the exact roster tokens and weights, the seven
canonical seeds, the game-id and per-game seed functions, the ordered
schedule, `corpus_setup_source(split)`, the corpus storage schema, and the
commit/reconciliation rule (a game is trainable only once trajectory +
metadata exist, verify, and a commit record lands; resume reconciles ids,
never duplicates a committed game, never exposes an orphan). Agent 2 makes
no new learning-design decisions.

## 2. Agent 2 — Synthetic Rule-Agent Corpus

**Status: PASS** — 26 / 26 completion gates true. Machine-readable record:
`reports/phase_8_data/agent_02_corpus_manifest.json` (manifest + generation +
storage + gates), `reports/phase_8_data/agent_02_corpus_audit.json` (the full
independent audit, determinism evidence, and the Agent 3 handoff), and
`reports/phase_8_data/agent_02_matchup_counts.csv` (300 rows: one per ordered
cell per split). The corpus itself lives at
`/Users/brandonwashington/Dev/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1/`
with its own `manifest.json`; §2.12 records why it is there and how a consumer
finds it.

`synthetic_warmstart_corpus_v1` is complete and frozen:

```text
ordered policy pairs                            100
train games                                  20,000   (200 / cell)
validation games                              4,000   ( 40 / cell)
test games                                    4,000   ( 40 / cell)
total games                                  28,000

recorded decisions                        7,260,310
selected decisions (sampler v1)           1,747,060
corpus content digest    c95c3545b07f2341e7efbc83c79e6342510dd973038b0f72e7eae013cff87d0d
metadata digest          1db0f02fe45b16f539f070b1e12d4fdd6f390fd0487180fe660af0f4d49c81bb
commit-index digest      32e8e18d1ca57ee555ed848851284f5938d4989ceb6c864f83ca4b9286c15db1
```

No neural model made a single corpus action. Every ply was chosen by a frozen
Phase 4 rule policy.

### 2.1 Prerequisite verification

Verified live, not quoted:

| Check | Required | Found |
|---|---|---|
| Agent 1 | `PASS`, all gates | status `PASS`, 18 / 18 gates true |
| Agent 1 contract digest | `7b6e7b27…58de` | recomputed live from `warmstart_contract.contract_document()`, matches |
| Frozen upstream | rules / engine / observation / model contract / action encoding / trajectory / bank / library / sampler / profile | `verify_frozen_upstream(include_library_digest=True)` returned no problems |
| Setup library | `7b8a6660…02777` | loaded live, matches |
| Phase 4 roster | exactly the 10 accepted policies at their accepted versions | `verify_teacher_roster()` and `verify_live_population()` both clean |
| Policy roster digest | — | `1fd8f44ab0999b56c589ad783a4c60d67367c2c2a08f11d7d38707de7b2f38a4` (tokens + roles + weights) |

Pre-existing suite, measured at commit `144baf4` **before any Agent 2 edit**:

```text
.venv/bin/python -m pytest tests -q
3501 passed, 3 skipped, 0 failed in 200.81s
```

That is Agent 1's post-edit total, as expected — Agent 1's work was already in
the tree. The three skips are the same pre-existing PASS-gated skips.

### 2.2 What was built

```text
stratego/training/rule_population.py    one logical game from one game id
stratego/training/corpus_commit.py      the crash-safe commit store
stratego/training/synthetic_corpus.py   schedule, resume, audits, manifest
scripts/run_phase8_agent02.py           the acceptance harness
scripts/relocate_phase8_corpus.py       verified corpus relocation (§2.12)
tests/training/test_synthetic_corpus.py 34 regressions
tests/training/test_corpus_resume.py    16 regressions
```

Everything else is reused: `stratego.engine` is the sole legality and
termination authority, the Phase 4 policies and their `build_policy_input` /
`decide_checked` path are untouched, `trajectory_v1` is unchanged (no schema
edit anywhere), the Phase 7 setup sources are entered through their frozen
`corpus_setup_source(split)` entry points, and the shard container is the
accepted Phase 6B byte layout, so `shard_writer.verify_shard` and
`directory_summary` read a finalized corpus unchanged.

### 2.3 Game content is a pure function of the game id

`play_corpus_game(game_id)` takes an identifier and nothing else that can vary.
From the id it derives the split (which selects the setup source), the
setup-source root seed, and the two rule-policy match seeds; per-ply randomness
stays on the frozen Phase 4 `derive_decision_seed(policy_seed, ply)` path.
Worker count, process partitioning, arrival order and resume boundary are not
inputs anywhere, so they cannot be outputs. Corpus games run under the frozen
`TRAINING_RULES` (battleless 100 / absolute 4,000) that Agent 1 froze.

The header stores `setup_family` as the source label
(`setup_library_v1_setup_sampler_v1_<split>`); per-game Phase 7 identity lives
in the metadata sidecar, exactly as Phase 7 Agent 5 designed it.

**What a stored decision means.** `trajectory_v1` stores one probability per
legal action, and a rule policy publishes no distribution. The record therefore
stores the *realized* decision — 1.0 on the action the policy actually chose,
0.0 elsewhere — which is exactly the Phase 8 policy target of common-contract
§16.1. It is deliberately not a claim about the policy's behaviour
distribution, and Phase 8 uses no importance ratio, so nothing consumes it as
one. The value slot carries a constant neutral `(1/3, 1/3, 1/3)` for the same
reason: Phase 8's value target is the final outcome, never a stored prediction.
Each decision's `collection_policy_version` names the acting rule policy's
`id@version`, so the corpus records who moved at every ply.

### 2.4 The crash-safe commit protocol (`warmstart_corpus_commit_v1`)

This closes the Phase 7 sidecar/trajectory crash window for the static corpus.
The rule is one sentence: **a game becomes visible only when its commit record
exists.** One file set belongs to one (split, segment, worker):

```text
<root>/<split>/shards/seg0000_w00_s0000.stgshard    trajectory payloads
<root>/<split>/metadata/seg0000_w00.meta.jsonl      one JSON line per game
<root>/<split>/journal/seg0000_w00.commit.jsonl     one JSON line per commit
```

Per game, in order: encode → compress → **decode the compressed bytes back and
check they rebuild this game** → validate the metadata against the record →
append trajectory bytes, flush → append metadata line, flush → append commit
line, flush. "Verifies" means the persisted bytes decode to this game, not that
they are the right length.

Recovery is truncation, not repair, because each commit record carries the two
file sizes *after* its own writes. `reconcile_corpus` drops a torn tail line
from the journal, truncates the metadata file and the last committed shard to
their recorded offsets, and removes any shard written entirely after the last
commit. A shard rolls over only *between* games, at a committed boundary, so a
closed shard never holds an uncommitted record. After reconciliation every
surviving byte belongs to a committed game, which is why the finalization
requirement of zero orphans is structural rather than hopeful. Uncommitted work
is discarded rather than repaired — safe precisely because a game's content is
a pure function of its identifier.

Resume is subtraction: `pending = scheduled − committed`. There is no cursor
file and no checkpoint.

### 2.5 Independent audit of the persisted bytes

Every audit re-derives what it checks from the frozen contract and reads the
corpus back from disk; none of them trusts the generator's bookkeeping.

```text
games audited                                28,000 / 28,000
decisions replayed through the engine     7,260,310
illegal actions                                   0   (counted, not inferred)
replayed-vs-stored legal-set mismatches           0
snapshot-path observation cross-checks      111,847
full provenance rebuilds                     28,000 / 28,000
per-game audit problems                           0
```

The replay audit rebuilds each game from its two stored setups and replays the
record's **action list** through the engine, regenerating the legal action set
at every ply and requiring it to equal the stored one, requiring the decision
record to name the same action the action list does, and requiring the terminal
result, reason and length to match the header. At four evenly spaced plies per
game it additionally rebuilds the position through the *snapshot* path
(`reconstruct_state`) and compares board, acting player, legal actions and the
`observation_v2_1_127ch` digest with the linear replay's, so the two
independent reconstruction routes must also agree.

The provenance audit rebuilds **both setups of all 28,000 games** from
`setup_provenance_v1` alone through the frozen Phase 7 rebuild path,
re-orients each for its player, and requires equality with the setup the
trajectory stores, plus fingerprint, split, family and base-id agreement — the
"preferred hard evidence" form, not the sampled fallback.

Commit integrity reconciles the three identity sets:

```text
committed ids   28,000        duplicate committed ids            0
metadata ids    28,000        duplicate metadata / payload ids    0
payload ids     28,000        orphan trajectory records           0
                              orphan metadata records             0
                              missing payloads / metadata         0
trajectory digest mismatches       0
metadata digest mismatches         0
payload decode failures            0
split placement violations         0
```

### 2.6 Schedule and split isolation

Committed ids compared against a freshly enumerated schedule:

```text
train        20,000 / 20,000 scheduled, 0 missing, 200 in every one of 100 cells
validation    4,000 /  4,000 scheduled, 0 missing,  40 in every one of 100 cells
test          4,000 /  4,000 scheduled, 0 missing,  40 in every one of 100 cells
unscheduled committed games        0
cells with the wrong count         0
```

No cell borrowed a game from another cell: the counts are scheduled, never the
outcome of sampling.

Split isolation is validated from setup provenance, not from the output
directory:

```text
distinct setup base ids   train 6,385   validation 800   test 800
base-id overlap           train/validation 0   train/test 0   validation/test 0
game ids in two splits    0
```

Validation and test each drew all 800 of their available bases; train drew
6,385 of its 6,400 (uniform sampling of 6,400 bases into 40,000 side-draws
leaves a handful untouched, as expected). All 16 Phase 7 families appear, the
most and least frequent separated by under 3%.

### 2.7 Determinism evidence

```text
isolated rebuild            200 games replayed from their identifier alone;
                            re-encoded payload digest == the digest the commit
                            journal recorded; metadata identical         0 problems

worker / order independence 8 games, serial 1x1 vs parallel 4x2 in reversed
                            enumeration order -> identical content digest

crash -> resume             a run interrupted between the metadata write and
                            the commit, then resumed with a different worker
                            count, finalizes to the same content digest,
                            metadata digest and commit-index digest as the
                            clean run
```

The regression suite additionally injects a crash at every stage of the commit
protocol — before the trajectory write, after it, after the metadata write, at
the commit flush boundary, at a shard rollover — and, separately, kills a real
child process with `os._exit` mid-write. In every case the interrupted corpus
exposes only committed games, resumes without regenerating a committed id, and
converges to the clean corpus's digest with zero orphans. Staged runs
(`limit` 1, 3 and 5 of 6) confirm the resume boundary itself changes nothing.

### 2.8 Corpus diagnostics

Diagnostics only. The schedule was not altered in response to any of them.

```text
terminal reason      flag_capture                 15,524
                     battleless_move_limit_draw    9,509
                     opponent_no_legal_move        2,960
                     both_no_legal_move_draw           7

result               red_win 9,342   blue_win 9,142   draw 9,516
mean plies           259.0   (train 258.8 / validation 261.6 / test 259.4)
zero-decision games  0
```

Color balance is even in every split (red 32.6–33.5%, blue 32.5–33.1%, draw
34.2–34.3%), which is what a schedule that plays every ordered pair in both
directions should produce. Game length spans 2 to 1,476 plies; the bulk sits
between 100 and 400.

Selected training decisions under `warmstart_decision_sampler_v1`
(≤ 64 per game):

```text
train        1,247,173     validation 249,963     test 249,924
total        1,747,060     mean 62.4 per game
```

### 2.9 Performance and storage

```text
generation      28,000 games, 10 workers x 4 chunks (40 file sets per split)
storage         /Users/brandonwashington/Dev/stratego_phase8/warmstart/
                    synthetic_warmstart_corpus_v1        (local, non-iCloud)
                shards        216,165,725 bytes  (120 files, 40 per split)
                metadata      117,144,888 bytes
                journals       16,140,367 bytes
                total            353.0 MB + 120 shard manifests
free space      18.0 GB of 494.4 GB on the storage volume at finalization
```

The corpus bytes are excluded from version control (`.gitignore`:
`data/warmstart/`) — they are production data reproducible from the frozen
contract, and the manifest and digests that describe them are tracked.

### 2.10 What Agent 2 did not do

No training example was built, no observation tensor was materialized for
storage, no target was computed, no model was constructed or run, no optimizer
step occurred. The sealed test split was written and structurally audited only
— replay, provenance, schema and integrity — which is the `structural_audit`
purpose Agent 1's `check_test_corpus_access` allows before Agent 7; no model
touched it, and no selection decision consulted it. The Phase 4 evaluation bank
was not used at all. No engine, policy, library, sampler or Phase 4 artifact
was modified.

### 2.11 Post-edit suite and completion gates

Steady-state run, after the relocation of §2.12 and against the final source
state:

```text
.venv/bin/python -m pytest tests -q
3551 passed, 3 skipped, 0 failed in 225.32s
```

3,501 pre-existing tests plus 50 new Agent 2 regressions, all green; the three
skips are the same pre-existing PASS-gated skips as before Phase 8. An earlier
run of the same harness, before the relocation and before the last three
regressions were added, reported 3,548 / 3 / 0 — the same suite plus those
three tests.

26 / 26 completion gates true (recorded in `agent_02_corpus_audit.json`):
28,000 scheduled games exact; 20k/4k/4k split counts exact; all 100 cells exact
in all three splits; no missing and no unscheduled games; zero duplicate
committed ids; zero orphan trajectories and zero orphan metadata; zero missing
payloads and zero missing metadata; zero digest mismatches; zero decode
failures; zero split-placement violations; zero base-id overlap; zero illegal
actions; zero legal-set mismatches; every committed game replayed; replay and
provenance audit clean; isolated rebuild exact; crash/resume converges;
worker and enumeration order independent; manifest and digests written; no
neural corpus actions; upstream unchanged; live population unchanged; full
suite green.

### 2.12 Corpus relocation and the storage-path incident

The corpus was generated under the repository's preferred path while the
repository sat in an iCloud-synced folder. During finalization the machine's
internal volume filled to 97%, macOS began evicting iCloud file contents to
reclaim space, and every subsequent read of an evicted file blocked on a
re-download — including the interpreter's own site-packages, which stalled
`import torch` for minutes at a time and interrupted the acceptance sequence.

This was an environmental storage-path problem, not a corpus defect. Nothing
below re-generated a single game; the machine-readable record is
`reports/phase_8_data/agent_02_relocation.json`.

**Relocation.** The corpus reached its current root in two stages, and was
verified at each one before anything was deleted:

```text
stage 1   repository (iCloud) -> /Volumes/Brandon_Washington/stratego_phase8/...
          copied with `ditto`, then every file's SHA-256 compared across both
          trees: 481 / 481 identical. Only then was the source removed.
          Performed with shell tools because the interpreter was unusable at
          the time; `scripts/relocate_phase8_corpus.py` is the equivalent
          copy-verify-then-remove path for future moves.

stage 2   external volume -> /Users/brandonwashington/Dev/stratego_phase8/...
          part of moving the whole project off iCloud to local storage.

verify    at the final root: 481 corpus files (plus a Finder `.DS_Store`,
          which nothing reads), and
              content digest        c95c3545…f87d0d   == accepted
              metadata digest       1db0f02f…4c81bb   == accepted
              commit-index digest   32e8e18d…6c15db1  == accepted
              28,000 committed / 28,000 metadata / 28,000 payloads
              0 orphans, 0 duplicates, 0 digest mismatches,
              0 decode failures, 0 split-placement violations
```

The full 28,000-game replay and provenance audit of §2.5 was then re-run
against the relocated bytes and is the audit reported there.

The three digests are built from game ids and payload/metadata digests only —
never from shard filenames, worker ids, segment numbers or paths — which is
exactly why a corpus can change volumes and remain the same corpus. A
regression pins that property (`test_relocating_a_corpus_preserves_every_digest`).

**Finding the corpus.** The root is now resolved by
`synthetic_corpus.default_corpus_root()`, first match wins:

```text
STRATEGO_WARMSTART_CORPUS_ROOT     explicit per-process override
data/warmstart_corpus_root.txt     the recorded redirect (tracked, one line)
data/warmstart/...                 the contract's preferred path
```

The pointer file currently reads
`/Users/brandonwashington/Dev/stratego_phase8/warmstart/synthetic_warmstart_corpus_v1`,
and `agent_02_corpus_manifest.json` records both the resolved root and which
rule chose it. Agents 3–7 should call `default_corpus_root()` rather than
assume a path. This is the redirect the common contract's §5 allows; the
manifest records the real location and free space, and no external volume is
claimed to have been tested.

**A bug the move exposed.** The acceptance harness recorded the corpus
manifest's location with `Path.relative_to(REPOSITORY_ROOT)`, which raises for
any root outside the repository. The first finalize run after the relocation
therefore crashed while assembling its artifacts — after the audit, the shard
manifests and the full suite had all already succeeded. Fixed by
`synthetic_corpus.repository_relative()`, which returns the relative path when
the target is inside the repository and the absolute path otherwise, with a
regression
(`test_reporting_paths_survive_a_corpus_outside_the_repository`). The steady-
state run in §2.11 is the re-run after that fix.

**Environment note.** The transfer to local storage stamped
`com.apple.quarantine` on 25,663 virtualenv files (source `sharingd`), and
macOS refuses to load ad-hoc-signed dylibs carrying that flag — `torch` failed
with "library load disallowed by system policy". Clearing the flag on the
project's own `.venv` restored the interpreter. This touched no repository
source and no corpus byte.

**Handoff to Agent 3** (in the artifact's `handoff_to_agent_3` block): the
corpus root and manifest, the three digests, `CorpusReader` (journal-backed
game index, `record(game_id)` → `trajectory_v1` record, `metadata(game_id)` →
synthetic sidecar, `commits[game_id]` → commit record),
`rule_population.play_corpus_game(game_id)` to rebuild any game in isolation,
`warmstart_contract.corpus_setup_source(split)` for split access, the frozen
policy-supervision weights, `warmstart_seed.selected_decision_indices` as the
decision-sampler contract, the 1,747,060 selected examples the sampler yields,
and the three audit entry points. Agent 3 makes no new corpus decisions.
