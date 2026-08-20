# Phase 12 — Search Engineering
## Sequence and Common Contract

## 1. Purpose

Phase 12 is a **rapid search-engineering phase**.

Its central question is:

> Can search convert the improved Agent 1C beliefs into better move choices and higher game-winning strength?

This phase deliberately returns to an engineering-first workflow. It is not a publication-style acceptance process and should not grow into one.

The priorities are:

1. working search;
2. useful decision signal;
3. winning-strength improvement;
4. wall-clock efficiency;
5. production integration.

Do not add long validation cycles, exhaustive ablations, large hyperparameter sweeps, sealed-test procedures, or unrelated audits unless required to make the implementation function correctly.

## 2. Project history must be preserved

Phase 11 remains formally closed as:

```text
phase11_final_classification = FAIL
phase12_was_not_authorized_by_phase11 = true
```

Phase 11B was an engineering sprint, not a retroactive repair of Phase 11.

Preserve unchanged:

- accepted Phase 9 checkpoint;
- Phase 10 artifacts;
- all Phase 11 contracts and Agent 1–7 reports;
- `phase11_system_v1`;
- the spent Phase 11 sealed test bank;
- the Phase 11 access ledger;
- the final Phase 11 `FAIL`;
- all Phase 11 sampler, safety, reproducibility, and runtime evidence;
- all Phase 11B checkpoints, reports, leaderboards, and data;
- the selected Agent 1C checkpoint.

Nothing in Phase 12 may delete, overwrite, or reinterpret those artifacts.

## 3. Phase 11B closure decision

Phase 11B stops after the already-running Agent 4 experiment completes.

Do not launch Phase 11B Agent 5.

Regardless of Agent 4's result, the selected Phase 12 engineering belief candidate is:

```text
Agent 1C
R_CE = 0.9460
top-1 = 0.2640
```

Carry forward these engineering lessons:

- dedicated supervised belief training mattered substantially;
- C1's learned representation is valuable and sample-efficient;
- larger external models were not automatically better at the current data scale;
- useful optima can occur early;
- evaluate frequently;
- optimize for improvement per unit wall-clock and compute.

Agent 1C is an **engineering replacement candidate for search**, not a scientifically validated replacement for the Phase 11 belief system.

## 4. Model roles in Phase 12

Use the accepted Phase 9 C1 as the move model:

```text
Accepted Phase 9 C1
    policy
    value
```

Use Agent 1C only as the improved belief provider:

```text
Agent 1C
    belief prediction
```

Do not assume Agent 1C's policy/value outputs remain equivalent to accepted Phase 9 after its final C1 block was fine-tuned.

Therefore, until separately demonstrated otherwise:

```text
Phase 9 C1 -> policy/value
Agent 1C   -> beliefs
```

Trunk sharing may be explored later only as a performance optimization if exact equivalence can be demonstrated cheaply.

## 5. Search belief providers

The same search engine must support interchangeable belief providers:

```text
remaining_count
original_phase11
agent1c
oracle
```

Definitions:

- `remaining_count`: count-based baseline;
- `original_phase11`: original neural belief head from the accepted Phase 9/Phase 11 stack;
- `agent1c`: selected Phase 11B engineering candidate;
- `oracle`: true hidden information, used only as an offline diagnostic upper bound.

The oracle must never be available through normal production play.

## 6. Search principle

The initial search should be deliberately simple:

```text
public state
    ↓
belief provider
    ↓
K complete hidden worlds
    ↓
Phase 9 root policy
    ↓
candidate root moves
    ↓
evaluate each move on the same K worlds
    ↓
short Phase 9 greedy-policy rollouts
    ↓
terminal outcome or Phase 9 leaf value
    ↓
average across worlds
    ↓
policy-regularized score
    ↓
chosen move
```

Important rules:

- sample hidden worlds once at the root;
- evaluate candidate actions using the same sampled worlds;
- the simulated acting player sees only legally available information;
- sampled hidden truth must not be directly exposed to the real root player;
- terminal game results override neural leaf value;
- Phase 9 C1 provides rollout policy and leaf value.

A scalar leaf value may be computed as:

```text
V = P(win) - P(loss)
```

## 7. Candidate root moves

Initial rule:

```text
if legal actions <= 8:
    evaluate all legal actions
else:
    evaluate the top 8 actions under the Phase 9 policy
```

The direct Phase 9 move must always be included.

## 8. Policy regularization

Use a modest direct-policy prior:

```text
S(a) = Q(a) + beta * log(pi(a) + epsilon)
```

Use one fixed modest `beta` initially. Do not perform a grid search in the first agents.

## 9. Initial search presets

### TINY

```text
worlds       8
root moves   up to 8
depth        4 plies
rollouts     1 per action/world
```

### SMALL

```text
worlds       16
root moves   up to 8
depth        6 plies
rollouts     1 per action/world
```

### MEDIUM

```text
worlds       32
root moves   up to 8
depth        8 plies
rollouts     1 per action/world
```

Only consider a larger configuration such as 64 worlds and depth 10–12 if MEDIUM already shows meaningful strength improvement at acceptable latency.

## 10. Engineering metrics

Every search experiment should report at least:

```text
belief provider
world count
candidate count
search depth
C1 forwards per move
move latency
positions/second
move-change rate vs direct C1
oracle agreement rate when available
W / D / L
effective win rate
EWR by opponent where applicable
EWR improvement per additional search second
```

No significance claim is required during this engineering phase.

## 11. Agent sequence

Run sequentially:

```text
Phase 11B Closure Agent
        ↓
Phase 12 Agent 1 — Search Core
        ↓
Phase 12 Agent 2 — Belief-to-Decision Diagnostic
        ↓
Phase 12 Agent 3 — First Search Match Test
        ↓
Phase 12 Agent 4 — Budget Scaling
        ↓
Phase 12 Agent 5 — Working Search Player
```

Agents must stop and report after their assigned task.

Do not automatically launch the next agent without review.

## 12. Conditional progression

```text
Agent 1
working search
    ↓
Agent 2
does better belief improve decisions?
    ↓
Agent 3
does search improve wins?
    │
    ├─ NO
    │    ↓
    │  fix search mechanics before scaling compute
    │
    └─ YES
         ↓
       Agent 4
       determine useful search budget
         ↓
       Agent 5
       productionize the best search player
```

Do not compensate for a bad search design by blindly increasing world count or depth.

## 13. Interpretation framework

Early comparison:

```text
direct C1 policy
search + remaining-count belief
search + original Phase 11 belief
search + Agent 1C belief
search + oracle belief
```

The oracle is diagnostic only.

If Agent1C search > old-belief search > count search and Agent1C search beats direct C1, the belief/search mechanism is likely useful.

If oracle search >> Agent1C search but Agent1C search is weak, belief quality may still be limiting.

If oracle search also fails to improve over direct C1, search mechanics are likely the primary bottleneck.

## 14. Phase 12 end state

The desired output is:

```text
phase12_search_candidate_v1
```

containing:

```text
accepted Phase 9 checkpoint identity
Agent 1C checkpoint identity
search algorithm version
belief provider
world count
candidate count
depth
policy regularization
latency
quick match results
known limitations
phase11_final_classification = FAIL
phase11b_selection = Agent1C
scientific_validation_status = not performed
oracle_available_in_production = false
```

The goal is a **working search-enhanced player**, not a scientific final claim.
