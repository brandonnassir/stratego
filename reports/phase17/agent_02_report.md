# Phase 17 — Agent 2 report
## Fixed-transition move self-play from Phase 9

```text
artifact                      phase17_agent02_report_v1
work package                  phase17
provisional run id            RUN-2026-A
written                       2026-08-27 (UTC)
baseline commit               25e52c3c3013c751e319e92a70e5f8a2fdff1cbc
consumed handoff              phase17_contract_handoff_v1 (ready_for_agents_2_3: true)
evidence_classification       ENGINEERING
scientific_validation_status  not performed
ready_for_tandem_integration  TRUE
```

Agent 2 built the move half of Phase 17 and verified it. It trained no
production run, ran no evaluation, and makes no strength claim. Every number
below was recomputed from live objects by `scripts/run_phase17_agent02.py`;
none was transcribed from a test log.

---

## 1. Headline

Five things came out of this.

1. **The Phase 9 start verifies exactly, twice over.** Both claimed digests
   reproduce, and the rebuilt model puts *bit-identical* policy, value and
   belief logits on fixed observations against a second, independent accepted
   entry point (`bind_behavior_snapshot`, the Phase 9 collection loader).
   Digest equality alone would not have proved the model was rebuilt the same
   way; this does.
2. **The Phase 16 in-flight rebind defect is fixed structurally, not
   procedurally.** There is no per-runner snapshot to go stale, because the
   whole population reads one mutable cell through a property. The forced
   test shows 120 post-rebind decisions in the *same six games*, every one
   carrying snapshot B's digest *and* B's actual legal distribution, while the
   120 pre-rebind rows still carry A's.
3. **Gate `G-M4a` holds exactly — 0.0, not "within tolerance".** Across 18
   cases (lengths 1 to 200, all three outcomes) the tailed recursion
   reproduces the accepted whole-game recursion with a maximum difference of
   `0.0` in deltas, advantages and W/D/L targets.
4. **The measured partial-emission divergence reproduces Agent 1's finding**
   (0.3005 vs 0.3092 in advantage on the same construction — the small gap is
   the bootstrap source, see §5). It is recorded per row as telemetry and is
   never a gate. `G-M4b` is not reinstated and no tolerance was weakened.
5. **One defect was found and fixed during verification**: a game whose
   *terminating* action is the one that reaches the transition budget is still
   seated when the window loop exits. Bootstrapping such a trace would have
   invented a continuation past a terminal position. It is now sealed at the
   window close and closes on its real outcome. `WindowResult.sealed_at_boundary`
   counts it, and a test asserts the path is exercised.

---

## 2. What was built

Nine modules, all additive, all inside `stratego/training/phase17/`:

```text
move_contract.py         constants, versions, seeds, the schedule horizon,
                         the identity plumbing, and the participant refusals
move_start.py            the exact Phase 9 loader and the fresh
                         optimizer / KL controller / EMA / schedule state
move_snapshot.py         the live current-policy cell, the seating that reads
                         it, and the categorical action sampler
move_loss.py             the objective, with a per-row value mask and no
                         belief term
move_trainer.py          the one-epoch update, the accepted advantage filter,
                         the KL controller, the EMA, and the refusals
transition_schema.py     phase17_move_transition_v1 and its validator
transition_targets.py    the tailed recursions, the seat-trace carry state,
                         gate G-M4a, and the divergence telemetry
transition_collector.py  the fixed-transition window collector
__init__.py              a package marker with no re-exports
```

`__init__.py` is deliberately empty of imports. Agents 2, 3 and 4 own disjoint
module families in this package; a package `__init__` that imported one
agent's modules would put them in every other agent's import closure, which is
exactly what the structural no-search gate measures.

### Reused, not rewritten

Imported unmodified and proven so by the import-closure test:

```text
stratego.engine.*                     rules, observation, legality, transition
stratego.training.phase9_collector    GameRunner, NeuralRequest, the lockstep loop
stratego.training.phase9_contract     gamma, lambdas, PPO clip, the filter, the
                                      KL controller constants, the recursions
stratego.training.phase9_behavior     BehaviorSnapshot, behavior_distribution,
                                      evaluate_observations, state_dict_digest
stratego.training.phase9_checkpoint   read/validate/rebuild of the start file
stratego.training.phase9_loss         behavior_kl_per_row, legal_entropy_per_row,
                                      validate_behavior_matrix,
                                      behavior_probability_matrix
stratego.training.phase9_trainer      KLController
stratego.training.phase16.schedules   power_law_learning_rate, annealed_entropy
stratego.training.phase16.trainer     WeightEMA
```

`stratego.training.phase9_loss.phase9_batch_loss` was **not** edited. §4 says
why a second objective exists.

---

## 3. The exact Phase 9 start

```text
path                     checkpoints/phase9/selfplay_c1_v1.pt
file_sha256              dfd698e5…10ea      reproduced
model_state_digest       f1df694d…cefd      reproduced
parameter_count          863,959            reproduced
candidate                C1
lineage                  behavior_snapshot B041, post-iteration 40
loader                   read_phase9_payload -> validate_phase9_payload
                         -> model_from_payload
```

The model-state digest is
`stratego.training.phase9_behavior.state_dict_digest` over a live module. The
*other* function of that name gives `f0994cf0…869e` on the same bytes; the
loader's own error message names it, so a future reader cannot mistake a
convention mismatch for corruption.

**Logit identity.** Against `bind_behavior_snapshot(…, "B041", "canonical")` on
8 fixed observations:

| head | max absolute difference |
|---|---|
| policy_logits | 0.0 |
| value_logits | 0.0 |
| belief_logits | 0.0 |

**Weights only.** The file carries `kl_beta = 0.2` (already at its ceiling),
`rl_iteration = 41`, `global_optimizer_step = 47,086`, and an entropy schedule
position of iteration 40 of 60. None is adopted. Phase 17 starts at:

```text
iteration                0 (next: 1)
optimizer moments        0 entries, fresh AdamW
kl beta                  0.005, controller history empty
ema                      0 updates, byte-identical to the raw weights
belief loss weight       0.0   (the accepted Phase 9 weight is 0.25)
```

`check_phase9_resume_identity` is deliberately not called: it authorizes a
Phase 9 *resume*, and Phase 17 is a new lineage.

---

## 4. The objective, and why there is a second one

The accepted objective could not express two Phase 17 requirements:

1. its value term is meaned over **every** row with no per-row loss mask —
   which is precisely why Phase 16 left partial emission off;
2. its belief weight is a module-level `0.25`.

So `phase17_batch_loss` is assembled from the accepted *components*:

```text
L = L_PPO + 0.5*L_value + beta*D_KL(pi_b || pi_theta) - c_H*H(pi_theta)
```

`masked_soft_value_loss` reduces to the accepted `soft_value_loss` exactly when
every weight is 1 (difference `0.0`, asserted by
`assert_value_loss_reduces` and by a test that also checks a zero-weight subset
equals the accepted loss over that subset).

**The belief head is structurally absent, not weighted to zero.**
`outputs.belief_logits` is never read, so the head receives no gradient at all.
Three tests check this: no gradient on `belief_output.*` after a real backward
pass, the weights do not move across an update, and the telemetry reports
`loss_belief = 0.0` with `belief_term_present = False`.

**Names kept apart.** `c_H * H` is an entropy **bonus**; `beta * D_KL(pi_b ||
pi_theta)` is a **forward** behavior-KL penalty; the paper has neither — it has
a reverse KL to a magnet policy. All three are logged under their own names.

---

## 5. Targets: `G-M4a`, and what replaced `G-M4b`

Both recursions are one function each with the tail supplied:

```text
advantage   A_t = delta_t + lambda_A * A_{t+1}
            delta_last = tail_value - v_last,  A beyond the tail = tail_advantage
W/D/L       Y_t = (1 - lambda_V) P_{t+1} + lambda_V Y_{t+1}
            Y_last = (1 - lambda_V) * tail_prediction + lambda_V * tail_target
```

Substituting the true terminal continuation — `tail_value = z`,
`tail_advantage = 0`, both W/D/L slots the one-hot outcome — recovers the
accepted walk. The one-hot substitution is what makes the W/D/L reduction
*exact* rather than approximate: `(1-lV)Z + lV Z = Z`.

**Gate `G-M4a` result: pass, at 0.0.**

| cases | lengths | outcomes | max delta | max advantage | max W/D/L | tolerance |
|---|---|---|---|---|---|---|
| 18 | 1, 2, 3, 12, 47, 200 | win / draw / loss | 0.0 | 0.0 | 0.0 | 1e-6 |

**`G-M4b` is retired and stays retired.** On Agent 1's own construction — 12
decisions, boundaries at 4 and 8, `red_win` — partial emission diverges from
the whole-game targets by:

```text
max advantage divergence     0.3005     (Agent 1's probe: 0.3092)
mean advantage divergence    0.0906
max W/D/L divergence         0.3370
final-window rows            exact, 0.0 on every one
```

The small difference from Agent 1's 0.3092 is not a discrepancy in the
recursion: Agent 1's probe bootstrapped from the *next decision's own stored
prediction*, and this implementation bootstraps from a prediction taken at the
boundary **state**. Both are "the stored boundary prediction"; they are
different positions, and the reduction check above is what proves the
recursions agree.

Recorded per row and never gated: `target_provenance`
(`terminal_z` | `boundary_bootstrap`), `boundary_target_divergence`,
`boundary_wdl_divergence`, `bootstrap_age_windows`.

**`bootstrap_age_windows`, stated precisely** — the schema says "how many
windows the row's trace had been carried when the row was emitted", and under
partial emission every row is emitted in the window it was collected, so the
literal reading is degenerately 0. It is implemented as *how many window
boundaries the containing seat-trace had already crossed when the row was
emitted* (0 for a trace that began in the current window). Both readings agree
on a trace's first window. This interpretation is recorded here as a decision,
not slipped in.

---

## 6. The forced in-flight rebind

The Phase 16 defect: `WindowCollector.rebind` reassigns the *collector's*
`participants` reference, but each in-flight `GameRunner` keeps the
`IterationParticipants` object it was constructed with. Resolution is already
per-ply — it just resolves against a stale object, so an in-flight game keeps
playing under the weights it was created with for as long as it lives.

The fix: `CurrentMovePolicy` is one mutable cell; `Phase17Seating.behavior` is
a **property** over it; one seating object is shared by the whole population.
There is no per-runner copy to propagate to.

Two further guards make a stale decision impossible rather than unlikely:

- `apply_neural` refuses a request whose snapshot is not the cell's *current
  object* (identity comparison, not a token);
- an un-applied `NeuralRequest` is dropped at the window boundary, so a
  request prepared under the old weights can never be applied under the new
  ones. The engine state is untouched and the next window rebuilds it.

Evidence (`agent_02_forced_rebind.json`):

```text
snapshot A digest              f1df694d…cefd
snapshot B digest              a0519d99…2918      (differs)
policy token                   unchanged (the token names the seat role)
games in flight across rebind  6
pre-rebind rows                120, all bound to A
post-rebind rows, same games   120, all bound to B, both colours
stored distribution vs B       max difference < 1e-5 on every checked row
stored distribution vs A       > 1e-3 on every checked row
```

The distribution check recomputes the legal softmax from B's and A's raw logits
on the row's own stored observation. This is a claim about numbers, not about
metadata.

---

## 7. The fixed-transition window

Four windows of 192 transitions, population 6, with an update and a rebind
after each (`agent_02_window_verification.json`):

| iteration | emitted | bootstrapped | terminal | finished | trained | steps | raw changed |
|---|---|---|---|---|---|---|---|
| 1 | 192 | 184 | 8 | 1 | 48 | 3 | yes |
| 2 | 192 | 190 | 2 | 1 | 48 | 3 | yes |
| 3 | 192 | 192 | 0 | 0 | 48 | 3 | yes |
| 4 | 192 | 192 | 0 | 0 | 48 | 3 | yes |

```text
exact budget every window      yes (192/192, four times)
duplicate transitions          0 of 768
action replay from (stored distribution, stored seed)   768 of 768
participant ledger holds       yes: 4 distinct raw model states, all held by
                               the cell; 0 rule/stress, 0 historical, 0 search
EMA ever acted                 no (12 updates, digest not among the 5 that acted)
```

`transitions_trained` is 48 of 192 — the accepted `tau = max(Q75(|A|), 0.01)`
filter retaining a quarter, as it should. Harvested and trained counts are
logged separately.

**Exactness is achieved by truncating the batch**, not by overshooting and
discarding: a forward batch that would exceed the budget is cut, and the games
left over keep their pending state and are decided first in the next window.

---

## 8. Bounded verification — what was run

Everything in this section is CPU, single process, on the operator's machine.

```text
Agent 2's 8 test modules          174 passed          28.4 s
scripts/run_phase17_agent02.py    identity            0.6 s
                                  invariant           0.01 s
                                  rebind              2.0 s
                                  window              5.9 s
total move-specific runtime       under 1 minute
```

Checks, mapped to the instruction's section 8 list:

| asked for | where |
|---|---|
| exact Phase 9 logit identity | `test_move_start.py`, `agent_02_start_identity.json` |
| fixed-transition and target invariants | `test_transition_targets.py`, `test_transition_collector.py` |
| forced in-flight rebind | `test_transition_collector.py`, `agent_02_forced_rebind.json` |
| both-seat current-policy ledger | `agent_02_window_verification.json` |
| stochastic legal sampling | `test_move_sampling.py` |
| one small update: raw moves, EMA follows without acting | `test_move_loss_and_trainer.py` |
| boundary-state save/load round trip | `test_transition_collector.py`, `test_transition_targets.py` |
| structural no-search | `test_move_no_search.py` |

No LR arm, no strength comparison and no opponent-mixture experiment was run,
per section 8.

**Test counts are scoped to this agent.** `tests/training/phase17/` is shared
with Agent 3, which worked in parallel; the 174 above are Agent 2's own eight
modules, named explicitly in `agent_02_test_results.json` so the setup half's
tests are not attributed here.

**Regression check.** The whole repository suite was run to completion during
this work: **7,318 passed, 3 skipped, 0 failed** in 11 min 9 s. That snapshot
was taken while Agent 3 was still adding files, so its total is not a stable
count of the repository; what it establishes is that nothing accepted broke.
A second run over everything *except* `tests/training/phase17/` came back
**7,028 passed, 3 skipped, 0 failed** — exactly the 7,031 collected before this
work began. See §13.

---

## 9. Defects found and fixed during this work

**1. A game terminating exactly at the budget was left seated (fixed).**
When the action that reaches the transition budget is also the action that
ends the game, the loop exits before the runner is advanced again, so the slot
still holds a finished position with an unsealed trajectory. The trace would
then have been closed on a bootstrap of a *terminal* state — a continuation
that does not exist. `_seal_terminal_slots` now finishes and retires such a
runner at the window close, before anything is emitted;
`WindowResult.sealed_at_boundary` counts it and
`test_the_budget_is_exact_across_a_short_and_long_game_mixture` asserts the
path is actually exercised. Found by running four windows rather than one.

**2. `bootstrap_age_windows` is under-specified in the schema (recorded, not
fixed).** See §5. The interpretation used is documented in the module, the
handoff and this report.

**3. Two small robustness fixes made after the first verification pass.**
`WindowResult.rule_decisions` was incremented once per colour rather than once
per runner (always 0 in practice, since the rule path raises). And
`assert_ema_never_acted` hashed the EMA by wrapping it in a throwaway module,
which required mangling the dotted parameter names; it now applies the
accepted `state_dict_digest` walk to the mapping directly, with a test
asserting the mapping and module digests agree on the same weights.

---

## 10. What Agent 2 did not establish

- **No setup network.** Every verification game drew its setups from an
  explicit Agent 2 **test double** (`phase17_agent02_test_double_v1`, uniform
  random). Nothing here measures setup quality, and the collector has **no
  default provider** — a missing one is refused, so the double cannot leak
  into a production path.
- **Game lengths here are meaningless.** Uniform-random setups put flags on
  the front row often; some verification games end in three plies. Nothing
  about game length, terminal reason or win rate in these artifacts is a
  measurement of anything.
- **Throughput here is not a preflight measurement.** CPU, single process,
  small populations. It must not be used to estimate `N`.
- **`N`, `n_ref` and the frozen schedule curve are not established.** Only the
  formulas and the `MoveScheduleHorizon` object are. Agent 4 measures and
  freezes the horizon.
- **No paired checkpoint.** `collector.state()` carries the seat traces and
  the seated game identities; engine-state persistence for in-flight games is
  Agent 4's, per common contract section 10.
- **No strength claim, no evaluation, no production run.**
- **The value row mask is built but not exercised in production.** Every row
  carries `value_row_weight = 1.0`, because section 6 requires training on
  exactly the configured budget. The mask exists so a later decision to
  down-weight bootstrapped rows needs no new objective.

---

## 11. Carry-forward for Agent 4

| id | title |
|---|---|
| A2-CF1 | Exact active-game persistence is still open — the move half carries traces and seated identities; engine states are Agent 4's |
| A2-CF2 | The per-row value mask defaults to 1.0 everywhere and is not exercised |
| A2-CF3 | Boundary predictions cost up to `2 x population` extra forward rows per window; the throughput rehearsal should count them |

The order of operations Agent 4 must follow is fixed and is in the handoff:

```text
collect_window  ->  train_window  ->  cell.rebind_from_model(model, iteration=n+1)
```

The rebind must happen **before** the next window, and must be given the RAW
model. The EMA never enters the cell.

---

## 12. Artifacts

```text
reports/phase17/agent_02_report.md                  this file
reports/phase17/agent_02_start_identity.json        start digests + logit identity
reports/phase17/agent_02_target_invariants.json     G-M4a, and the divergence
reports/phase17/agent_02_forced_rebind.json         the rebind, on numbers
reports/phase17/agent_02_window_verification.json   4 windows, ledger, EMA, carry
reports/phase17/agent_02_test_results.json          the test-suite result
reports/phase17/phase17_move_handoff_v1.json        the handoff
scripts/run_phase17_agent02.py                      regenerates every number above
```

```text
gates            G-M4a pass | G-M4b retired | C2 pass | C3 pass
                 C4 pass | C5 pass | C6 pass | C9 pass
ready_for_tandem_integration   TRUE
```

---

## 13. Working-tree state

Nothing tracked was modified. The three tracked files that were already
modified before Phase 17 began — `reports/phase13/phase14_launch_manifest_v1.json`,
`stratego_project_docs/05_project_plan.md`, `stratego_project_docs/README.md` —
are untouched and remain unstaged, per the Agent 1 baseline rule. Nothing was
committed, stashed, cleaned or checked out.

Everything added is untracked and additive:

```text
stratego/training/phase17/move_*.py
stratego/training/phase17/transition_*.py
tests/training/phase17/test_move_*.py
tests/training/phase17/test_transition_*.py
scripts/run_phase17_agent02.py
reports/phase17/agent_02_*
reports/phase17/phase17_move_handoff_v1.json
```

One file is shared and was **merged, not clobbered**:
`stratego/training/phase17/__init__.py`. Agent 3 worked in parallel and had
written a docstring describing the setup half; the move half's module list was
added alongside it and Agent 3's text was preserved. The module still imports
nothing, which is the property the structural no-search gate depends on.
`tests/training/phase17/__init__.py` and `tests/training/phase17/conftest.py`
are Agent 3's and were left alone.

### Regression evidence

```text
everything except tests/training/phase17   7,028 passed, 3 skipped, 0 failed, 10m01s
full repository suite (mid-work snapshot)  7,318 passed, 3 skipped, 0 failed, 11m09s
Agent 2's own 8 test modules                 174 passed, 28.4 s
```

The 7,028 + 3 is exactly the count collected before this work began (7,031),
so the accepted suite is unchanged: Agent 2 added tests and broke none.
