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
