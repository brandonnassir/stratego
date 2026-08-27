# Phase 17 — Agent 1 report
## Contract, paper map, immutable baseline, and shared schemas

```text
artifact                      phase17_agent01_report_v1
work package                  phase17
provisional run id            RUN-2026-A
written                       2026-08-27 (UTC)
commit at time of work        124f3beed4fe4cfb02021de0185a8d871feec1c1 (main)
evidence_classification       PENDING
scientific_validation_status  not performed
ready_for_agents_2_3          FALSE — one blocking operator decision (D1)
```

Agent 1 trained nothing, ran no game, ran no gate, and committed nothing. It
read, recomputed, and froze. Everything below that is a number was recomputed
from bytes on disk by `scripts/run_phase17_agent01.py`.

---

## 1. Headline

Five things came out of this that change what the next agents should do.

1. **The Phase 9 start verifies exactly.** Both claimed digests reproduce. But
   the repository contains **two different functions named `state_dict_digest`**
   that disagree on these bytes, so Phase 17 now names one of them.
2. **Formal Phase 14 closure is already unreachable** — and not because of
   anything Phase 17 did. `--role finalize` refuses today. "Wait until the run
   is formally closed" therefore has no defined end, which matters because that
   is the condition the repository freeze is written against.
3. **Common-contract section 6 asks for two things that cannot both be true.**
   Partial emission with boundary bootstrapping, *and* windowed targets equal to
   whole-game targets. Measured divergence: up to **0.309** in advantage and
   **0.121** in W/D/L.
4. **The Phase 16 current-policy defect is real and located.** An in-flight game
   resolves every ply through the participants object captured at game creation.
   `rebind()` cannot reach it.
5. **The evaluation instrument cannot resolve what section 14 asks it to rank.**
   25 candidates on a 120-game lane produce **≈0.1435 EWR of apparent spread from
   pure noise alone**.

---

## 2. Process and source boundary

`reports/phase17/agent_01_process_boundary.json`

Read first, and read completely: `STATUS.md`, `PHASE_HISTORY.md`,
`EVIDENCE_INDEX.md`, `EXPERIMENT_FRAMEWORK.md`.

| Question | Observed |
|---|---|
| Learner, collector, supervisor or evaluator running? | **None.** |
| Anything running at all? | One process: `scripts/phase14_dashboard.py --port 8714`, pid 99915, ~3 d 18 h, HTTP 200. A read-only monitor. **Left running.** |
| Repository / run freeze active? | **Yes** — `STATUS.md` §5: do not commit, stash, checkout or clean until Phase 14 is formally closed. |
| Phase 14 run state | `closed: false`, `closed_reason: "emergency stop"`, 102 iterations, 59.97 h. Closure deadline `2026-08-28T16:15:34.689Z`. Read read-only. |
| Phase 16 still untracked? | **Yes.** 36 untracked entries. |
| Any accepted checkpoint or result path mutated? | **No.** The one tracked modification under an accepted path is `reports/phase13/phase14_launch_manifest_v1.json`, the known self-referential rebuild. Agent 1 did not touch it. |

I did not signal, stop, or restart anything. The dashboard was queried twice
with `Connection: close` and answered 200 both times; it must **not** be opened
from a browser, because a pooled keep-alive socket wedges it permanently.

### 2.1 Finding F1 — Phase 14 closure is already blocked

`scripts/phase14_launch.py` calls `assert_bound_launch_code()` **before** it
branches on `--role`, so the closeout path is gated by the same code binding as
a launch. Run read-only today:

```text
Phase14LaunchError: Phase 14 may not launch on this code revision:
  - tracked working-tree state differs from the manifest:
      bound    ['reports/phase13/phase14_launch_manifest_v1.json']
      observed ['reports/phase13/phase14_launch_manifest_v1.json',
                'stratego_project_docs/05_project_plan.md',
                'stratego_project_docs/README.md']
```

`code_digest` matches and `revision` matches. The refusal is purely the dirty
list, and the two extra files landed with the **2026-08-27 documentation pass**.

This is not a Phase 17 problem, and I did not fix it: rebuilding the manifest is
a Phase 13/14 action. But it changes the shape of decision D1, because the
freeze's release condition currently cannot occur.

---

## 3. The Phase 9 start, recomputed

`reports/phase17/phase17_start_identity_v1.json`

Loaded through the accepted path — `read_phase9_payload` →
`validate_phase9_payload` → `model_from_payload`. `check_phase9_resume_identity`
was deliberately **not** used: Phase 17 is a new lineage, not a resume.

```text
path                 checkpoints/phase9/selfplay_c1_v1.pt
file sha256          dfd698e5b6cf536a523bdd35dd3a32a513c2d83fb7c3936524d59786179b10ea   MATCHES
model state digest   f1df694d59e3435994be06f2537d9c603749bc072fc39bf021aac79f2dffcefd   MATCHES
parameters           863,959                                                            MATCHES
lineage              behavior_snapshot B041, produced after iteration 40
rules                stratego_project_v1
observation          observation_v2_1_127ch
action encoding      source_destination_10000_v1
ema_state in file    null
```

`phase14_contract`'s four bindings all agree with the recomputed values.

### 3.1 Two functions, one name

This is the one hazard in the start identity, and it is now frozen rather than
left to be discovered:

| function | input | hashes | value on these bytes |
|---|---|---|---|
| `stratego.training.phase9_behavior.state_dict_digest` | live `nn.Module` | name + shape + **float32 bytes** | `f1df694d…` |
| `stratego.model.checkpoint.state_dict_digest` | `state_dict` mapping | name + shape + **dtype** + bytes | `f0994cf0…` |

`f1df694d…` is the digest every accepted contract from Phase 9 Agent 8 onward
uses, so **that one is `phase17_model_state_digest`**. `f0994cf0…` is also what
sits in `model_state.provenance.state_dict_digest`, so a future agent comparing
against the provenance field will get a mismatch that is a naming error, not
corruption. Both values are recorded.

### 3.2 Start semantics frozen

Weights only. Fresh AdamW moments, LR schedule reset to iteration 1, fresh move
KL controller at β₀ = 0.005, fresh move EMA seeded from the loaded **raw**
weights. Setup model, optimizer, controller and EMA from scratch. The belief
head is present and takes **loss weight 0.0**, overriding the accepted Phase 9
`BELIEF_LOSS_WEIGHT = 0.25`.

---

## 4. The Ataraxos map

`reports/phase17/ataraxos_method_map_v1.md` · `…_v1.json` — 38 rows.

| status | rows |
|---|---|
| `exact` | 19 |
| `scaled` | 6 |
| `intentional divergence` | 9 |
| `not used` | 4 |

I read the paper itself (`2511.07312v1.pdf`, 46 pages), not a summary of it —
§2.1–2.6, §3, and appendices B, C, D.1–D.7 and E.

**One derived constant carries the rest:**

```text
N_paper = 8.56e6 move gradient steps / 202 batches per iteration = 42,376
```

Table 27 gives the gradient steps; D.4 gives 202 batches at one epoch. Every
re-horizoning argument below is arithmetic against that number.

### 4.1 Three regularizers that must never be called each other

This is where a summary would have produced a false result, so it is worth
being blunt.

| | direction | coefficient | in Phase 17? |
|---|---|---|---|
| paper move behaviour KL | **reverse**, `KL(π_θ ‖ π_θt)` | fixed `0.1` | no |
| accepted Phase 9 behaviour KL | **forward**, `D_KL(π_b ‖ π_θ)` | adaptive β, target 0.015 | **yes** |
| paper magnet KL | reverse, toward a uniform-piece-then-uniform-move policy | `0.05·n^-0.3` | no |
| Phase 17 entropy bonus | — (entropy, not a KL) | `max(0.001, 0.005·n^-0.3)` | **yes** |

The `n^-0.3` shape is the *only* thing the last two share. An entropy bonus
pulls toward uniform over legal actions; the magnet KL pulls toward a
structured distribution that is not uniform over legal actions. Rows M08 and
M09 record this; the telemetry contract forbids either field being labelled as
the other.

### 4.2 The move LR is flatter and shallower than the paper's

Recorded, **not changed** — common contract §9 is an operator instruction and it
wins.

```text
paper       clip(0.5·n^-1.1, 5e-6, 1e-4)
            ceiling held to  5.44% of the run
            floor reached at 82.86%
            dynamic range    20x

Phase 17    clamp(1.5e-4·(n/n_ref)^-1.1, 1.5e-5, 1.5e-4), n_ref = ceil(0.125N)
            ceiling held to 12.50% of the run
            floor reached at 101.39%  →  never, within the run
            dynamic range    10x
```

A shape-preserving map would be `n_ref = ceil(0.0545·N)` with
`lr_min = lr_max/20 = 7.5e-6`. Flagged as **D3-adjacent** context; **do not
amend without the operator**.

Related, and quantified because doubling the horizon changed it: the move
entropy floor `0.001` is reached at `n = 214`. On a 6-hour run (N≈313) that is
68% of the way through. On a **12-hour** run (N≈626) it is **34%** — so roughly
two thirds of the production run would sit at the terminal floor, which is
precisely the Phase 14 failure mode the Phase 16 schedule module was written to
avoid.

### 4.3 The setup architecture arithmetic settles FF width

`0.8M` and feed-forward 512 are consistent; 51 is not.

```text
4 blocks x (attention 66,048 + feed-forward 131,712 + 2 layernorms 512)  793,088
+ token embeddings 13 x 128                                                1,664
+ positional 41 x 128                                                      5,248
+ final norm                                                                 256
+ heads: next-piece 1,548 · W/D/L 387 · entropy 129                         2,064
                                                                        --------
                                                                         802,320
```

With feed-forward 51 the same arithmetic gives **328,412**. 512 also preserves
the paper's 4× width ratio (2,048 at width 512). **512 governs.**

### 4.4 Two places the paper is ambiguous, and how they are frozen

**Setup advantage units (D4).** As printed, Eq. (1) regresses `h` toward `H/10`
while `δ` uses `(H − h)`. Those are different units. The literal form
degenerates to roughly `0.9·α·H`, an *uncentered* bonus: with early setup
entropy around 69 nats and α = 0.1 that term is ≈ 6.2, against an outcome term
bounded by 2. Frozen as `α·(I/10 − h_θt)`, both sides in the units the
prediction loss actually trains, where `I = −log π_θt(σ̄|σ)` in nats.

**Setup entropy horizon (D3).** Raw transcription of `0.1·n^-0.3` onto a
12-hour horizon ends at α = 0.014489 against the paper's 0.004091 — **3.54×
more heavily regularized at the end than the paper ever was**. Frozen as the
endpoint-preserving exponent rescale:

```text
alpha(n) = max( 0.1 · n^-p , 0.004091 )      p = 0.3 · ln(42376) / ln(N)
alpha(1) = 0.100000   alpha(N) = 0.004091    exactly, for any N
```

The move LR's `n_ref` shift was not reused here because it preserves only the
upper endpoint. A re-horizoning that moves an endpoint is not one. Recorded
risk: the paper warns that annealing too aggressively "collapsed the entropy of
the model", and traversing a week-long anneal in 12 hours is aggressive by
construction — which is why the setup-entropy stop conditions exist.

---

## 5. Two defects found by reading the code

### 5.1 The current-policy defect is real, and `rebind()` cannot fix it

Common contract §5 names it a hard blocker. It is:

- `stratego/training/phase16/collector.py:536` — `rebind()` replaces the
  **collector's** `participants` and refuses only if the *logical identity*
  moves; a same-identity weight change is accepted silently.
- `stratego/training/phase16/collector.py:516` — a runner is constructed with
  `self.participants` **at game creation**.
- `stratego/training/phase9_collector.py:332` — the runner stores that object.
- `stratego/training/phase9_collector.py:447` — every ply resolves
  `acting_snapshot_for(self.scheduled, self.participants, actor)` from the
  runner's own copy.

Snapshots are frozen (`assert_frozen`), so an in-flight game keeps its
game-start weights for its entire length. Verified by source reading and by the
collector's own docstring, which states the intent plainly. It was **not**
verified by executing a live rebind — that is Agent 2's gate **C5**, and it
should be written before the fix, not after.

### 5.2 Section 6 asks for two incompatible things

`reports/phase17/agent_01_boundary_target_probe.json`

Phase 16's `window_edge_invariant` returns exactly `0.0`. It is exact because
`windowed_targets` **buffers the whole game** and computes at the close;
`partial_advantages` is called at each boundary and its result is discarded into
`boundary_reports`. So that invariant measures the *reduction* property, not
partial emission.

Running the path §6 actually mandates — three windows, bootstrapped tails — on a
12-decision synthetic track:

```text
max |A_bootstrapped − A_wholegame|   0.309161
mean absolute difference             0.083371
rows over the 1e-6 tolerance         7 of 12
max W/D/L difference, window 1       0.121119
```

Only the final window, whose tail is the true terminal `z`, reproduces the
accepted targets. This is not an implementation defect; a truncated λ-return
closed on a value estimate is not the full λ-return.

There is a further practical consequence for Agent 2: `phase9_batch_loss`
averages the value and belief terms over **every** row and has no per-row loss
mask, which is exactly why Phase 16 left partial emission off. Phase 17 needs a
**new** phase17-namespace target and loss path. The accepted objective must not
be edited.

Gate **G-M4a** (the reduction invariant) is the satisfiable form and is frozen
as final. Gate **G-M4b** (literal equality) is recorded as `not_run` with
`values_final: false` and is decision **D2**.

---

## 6. What is frozen

`reports/phase17/phase17_contract_handoff_v1.json`

- **Schemas** — `phase17_move_transition_v1`, `phase17_setup_episode_v1`,
  `phase17_joint_checkpoint_v1`, `phase17_eval_bundle_v1`,
  `phase17_eval_receipt_v1`. Field by field, with the digest rules, the
  atomic-write rule and the fail-closed compatibility rule. Delivered **in the
  handoff, not as `stratego/training/phase17/contract.py`**, because §4 permits
  writing that module only once the source baseline is immutable — and it is
  not. Agent 2 encodes them.
- **Move schedule and controller** — final, exactly as §9 states, with the
  deviations recorded rather than applied.
- **Setup schedules** — LR 5e-5 constant and the loss coefficients transcribed
  exactly; α re-horizoned as above; the setup KL controller and the episode
  queue explicitly **provisional** pending Agent 3's soak.
- **Composite benchmark** — `phase17_composite_benchmark_v1`, two lanes, the
  accepted `ebd13019…` pack as the move-only lane and a 120-case fixed-seed
  joint lane. Semantics frozen; the digest is computed when Agent 5
  materializes it. The **decision instrument is named in advance**: both full
  120-case lanes. `quick60` and `joint_quick60` are secondary readings and may
  not overturn it — Phase 16's quick-60 and full-120 reversed each other on the
  same games.
- **Gates** — G-C, G-M4a, G-M4b, G-S, G-E, G-W, with `pass | fail | not_run`
  and the rule that absence never means pass.
- **Stop policy** — 7 immediate conditions, 8 persistent-collapse conditions,
  each with its consecutive count and whether its threshold is final.

### 6.1 The instrument cannot resolve what §14 asks it to rank

Computed before any margin was set, per `EXPERIMENT_FRAMEWORK` §3.3:

```text
SE, one lane, 120 games, p≈0.80                              0.0365
SE, both lanes, 240 games, p≈0.80                            0.0258
SE, one opponent stratum, 12 games, p≈0.80                   0.1155
expected apparent spread over 25 candidates, pure noise       0.1435
expected worst-of-10 strata under pure noise at p=0.80        0.6222
stop threshold 0.15 below hour 0                            4.11 SE
```

So: picking the single highest mean EWR from 25 candidates is largely selecting
noise, and a *worst-stratum ranking* across 25 candidates is close to pure
noise. The 0.15 collapse threshold, needing three consecutive readings, is a
sound collapse detector — it just is not a selector.

Recommended (**D6**, and I have **not** changed §14): make the three-point
rolling median the primary direction instrument, use worst stratum as an
absolute floor filter with the floor set below 0.622, and re-evaluate the top
two or three shortlisted candidates at a materially larger paired game count
before promotion. Phase 14 is the precedent: its 128-game headline reordered
against a 2,200-game re-evaluation.

---

## 7. The baseline, and the one blocking decision

`reports/phase17/agent_01_baseline_inclusion_list.json`

```text
include   35 paths · 260 files · 39.4 MiB
          all Phase 15/16/17 source, tests, evidence, instructions, scripts,
          the four canonical status documents, data/phase16, and the two
          Phase 16 candidate JSONs
exclude   22.44 GiB via new .gitignore rules, in the pattern already used for
          phases 8, 9, 10, 10b, 11, 11b, 13 and 14:
            checkpoints/phase15/        15 GB
            data/phase15/              7.5 GB
            checkpoints/phase16/arms/   49 MB
leave     all three modified tracked files unstaged
```

I did not commit, and I ran no `git clean`, `stash`, `checkout` or `reset`.

**D1 — the operator's call:**

| option | effect |
|---|---|
| A · wait for formal closure | Agents 2–4 stay blocked, and per finding F1 the release condition currently cannot occur |
| **B · commit the untracked include list only** *(recommended)* | HEAD moves off `124f3be`; `assert_bound_launch_code` already refuses today on the dirty list alone, so this adds a second already-failing check rather than breaking something that works |
| C · rebuild the Phase 14 launch manifest first | restores the code binding, then decide; a Phase 13/14 action I did not take |

`ready_for_agents_2_3` is **false** until D1 lands. The schemas themselves are
complete — the block is the baseline, not the specification.

---

## 8. Commands run

```bash
.venv/bin/python scripts/run_phase17_agent01.py --role all
.venv/bin/python scripts/run_phase17_agent01.py --role bind
.venv/bin/python -m pytest --collect-only -q
```

`--role all` runs `observe`, `identity`, `probe`, `methodmap`, `inclusion`;
`--role bind` re-verifies every artifact digest against bytes on disk. Total
runtime under a minute; nothing here is compute-bound. `PYTHONPATH` must include
the repository root, and `python` must be `.venv/bin/python` — the pyenv shim
has no torch.

---

## 9. What Agent 1 did not establish

- **No gate was run.** Every gate result is `not_run`.
- No training, collection, evaluation or game was executed.
- `N`, `n_ref` and `p_setup` are **not** frozen — only the formulas are. Agent 4
  measures `N` in preflight. The ~626-iteration figure quoted anywhere in these
  artifacts is a Phase 16 extrapolation and is explicitly not a commitment:
  Phase 17 has two neural seats where Phase 16 had rule and stress opponents, so
  real throughput may be materially lower.
- The composite pack was not materialized and has no digest.
- The MacBook was not contacted. Nothing about its capacity, transport or
  cadence latency is established, and G-E4's p95 must be **measured**, not
  estimated, before unattended operation.
- The setup network was not built. 802,320 parameters is arithmetic.
- The full suite was not run. **7,031 tests collect cleanly** at `124f3be`. I
  changed no library code, so no suite regression can come from this work — but
  I am not claiming a suite pass.
- The current-policy defect was established by source reading, not by executing
  a live rebind. Gate C5 is the executable proof.
