# Phase 16 — Robustness and Distribution
## Overview, goal, constraints, and the parallel plan

_Written 2026-08-25. Approved by the operator after review of the Ataraxos gap report
(arXiv 2511.07312 vs phases 9–15). This file governs every Phase 16 agent; each agent
also has its own numbered instruction file in this directory._

## 1. The recalibrated goal

The prior "85% vs a typical human" target is retired: no pool of skilled humans is
available to measure it. The operator is the referee.

```text
phase16_goal_v1: the maximum-strength configured player achieves
EWR >= 0.50 (draws count half) over a predeclared 20-game exam
against the operator, under rematch conditions — the operator may
use any legal setups and adapt freely across games.
```

The operator has stated how the current system loses to them:

1. **adverse setups** — setups outside the training/search distribution;
2. **predictable decisions** — the deployed player is deterministic end-to-end
   (argmax move selection over greedy rollouts), so repeated play teaches its habits.

These two mechanisms are the phase's targets. They match the evidence already in the
repository (deep-search pilot: oracle worlds scale while belief-sampled worlds
anti-scale; Phase 14: 6-hour plateau + anchor specialization; paper: fixed-setup
ablation flattens learning; removing the search's policy-KL anchor costs 485 Elo by
"overfitting to idiosyncrasies… that fail to generalize to other opponents").

## 2. Phase 14 disposition — decided

- The run was **stopped 2026-08-24T04:19Z** at 59.97 h / step 202,504. Nothing is running.
- The operator has decided it will **not be resumed**.
- There is **no supported early close**: `finalize` refuses before the immutable
  deadline `2026-08-28T16:15:34.689Z`. Formal closure is Agent 5's first task, and it
  happens **only after** that instant.
- Until closure, the repository freeze holds:

```text
do not edit ANY tracked file          (breaks assert_bound_launch_code)
do not commit, stash, or checkout     (the manifest must stay the only tracked diff)
do not touch Phase 14 run state, checkpoints, logs, or control files
never run git clean
```

All Phase 16 work is **additive and untracked** until Agent 5 lifts the freeze.

## 3. Immutability rule (operator directive)

**Nothing already done is altered.** Accepted phase namespaces (2–15) are read-only.
Two legal moves only:

1. **reuse by import** — call accepted modules unmodified;
2. **rebuild in `phase16` namespaces** — when behavior must differ, write new code in
   `stratego/<area>/phase16/`, `tests/<area>/phase16/`, `scripts/*phase16*`,
   `data/phase16/`, `checkpoints/phase16/`, `reports/phase16/`.

Never copy-and-tweak accepted files in place, never monkey-patch accepted modules,
never overwrite accepted artifacts or checkpoints. Frozen model bytes are consumed
through their existing read-only copies (`checkpoints/phase15/p24_source_readonly.pt`
etc.) and verified by digest before use.

## 4. The agents and what runs in parallel

| Agent | Charter | Compute | May start |
|---|---|---|---|
| **1** Measurement & operator exam | untracked backup; canonical benchmark pack; adversarial setup pack + baseline; operator protocol and logging | match packs (hours) | **now** |
| **2** Stochastic search | sampled move selection + sampled rollouts; predictability diagnostics; temperature selection; `play_phase16.py` | position packs + match packs (hours) | **now**, parallel with Agent 1 |
| **3** Training loop v2 | window-based collector, damped schedules, EMA, pure self-play; the 3×6 h recipe shootout | 3 × 6 h training + eval | build **now**; 6 h runs under the compute lock |
| **4** Joint belief worlds | autoregressive hidden-army model; world-set metrics; provider; deep-ladder rerun | CPU training (minutes); ladder pack (~10 h) later | build **now**; ladder in Wave 2 |
| **5** Closeout & production | Phase 14 finalize; commits; housekeeping; production run; the operator exam | long run | **not before 2026-08-28T16:15:34Z**, then gated on Agents 1–4 |

Agents 1–4 are **code-parallel**: their namespaces are disjoint and none edits shared
state. Cross-agent consumption is by **frozen handoff artifact only** (a JSON binding
digests, like `phase15_search_handoff_v1.json`) — never by importing another agent's
work-in-progress.

Declared handoffs:

```text
agent1 -> agents 2,3,5 : phase16_measurement_handoff_v1  (benchmark + adversarial pack)
agent2 -> agents 1,5   : phase16_stochastic_candidate_v1 (selected play configuration)
agent3 -> agent 5      : phase16_recipe_candidate_v1     (winning training recipe)
agent4 -> agents 2,5   : phase16_belief_ar_candidate_v1  (world provider, if promoted)
```

If a needed handoff has not landed, use the declared fallback in your instruction file
(each agent has one); do not block and do not reach into another agent's namespace.

## 5. Compute coordination (one 14-core machine)

- **Heavy compute** = any job over ~10 minutes or more than 2 worker processes
  (match packs, training runs, corpus generation).
- Before launching heavy compute, create `checkpoints/phase16/COMPUTE_LOCK.json`:
  `{agent, task, started_utc, expected_hours, pid}`. Delete it when done. If the file
  exists and its pid is alive, wait or negotiate with the operator — do not co-run
  two heavy jobs.
- **Latency caps and any latency claim come from idle, single-process runs only.**
  Pack numbers are ~1.8× inflated by contention (Phase 15 measured it).
- Operator play sessions get an idle machine: no heavy compute while the operator
  is playing exam or baseline games.

## 6. Shared measurement conventions

- **EWR** counts draws as half. **Every reported EWR names its pack and version.**
  Cross-pack comparisons are forbidden in conclusions.
- Canonical instrument: `phase16_benchmark_v1` (Agent 1). Adversarial instrument:
  `phase16_adversarial_setups_v1` (Agent 1).
- Engineering margins, consistent with Phase 15: **0.10** selection margin;
  **0.03–0.05** meaningful band. These are engineering packs: predeclare every
  decision rule, report SEs, make no significance claims.
- Paired designs on shared boards and seeds wherever two arms are compared. Keep the
  `remaining_count` control in any belief comparison and the oracle as an offline
  ceiling (production refuses it by name — preserve the Phase 15 refusals).
- Every board that reaches `create_game` passes the accepted orientation gate
  (`red engine row == canonical rank; blue engine row == 9 - canonical rank`).
  Import Phase 15's section-4 gate; never re-derive it.
- Seeds derive from the phase namespace: `phase16.agentN.<task>` through the accepted
  seed-derivation helpers. Fixed seeds; sampled behavior must be reproducible from
  the seed.
- Reports: `reports/phase16/agent_0N_report.md` + machine-readable
  `agent_0N_*.json|csv`, appended per the repo convention. State plainly what was
  and was not established.

## 7. Non-goals — do not spend the phase on these

- **No anchor/EWR reward shaping.** Training reward stays pure W/D/L. Handcrafted
  bots and the anchor are evaluation opponents only (Phase 14 proved training on
  them buys specialization, not strength).
- **No search in the training loop** (the paper tried it and rejected it at a far
  better compute ratio than ours).
- **No scaling of the marginal belief head** (B-family ≈ count baseline in match
  play; the oracle ceiling at shippable budgets is +0.10–0.15).
- **No deeper-search rungs** before Agent 4's ladder gate passes (LARGE/XLARGE
  regressed −0.075 with current worlds; the ladder is closed until the worlds change).
- **No new sealed statistical test** (the Phase 11 bank is spent; Phase 16 is an
  engineering phase end to end).

## 8. Suite discipline

The full pytest suite stays green (6,708 passed / 3 skipped at phase start; only
additions allowed). Each agent adds tests under its `tests/*/phase16/` namespace and
runs the full suite before writing its report. Never mutate `os.environ` in code that
can run in the caller's process (Phase 15 defect 0).
